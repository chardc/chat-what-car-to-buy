import logging
import json
import datetime as dt
import pyarrow as pa, pyarrow.parquet as pq
from collections.abc import Generator, Callable
from typing import Tuple, NamedTuple, Union, Optional, Dict, List
from usedcaranalytics.pipeline.transformer import DataTransformer

logger = logging.getLogger(__name__)

class ParquetDataLoader:
    def __init__(
        self, config: Union[List[NamedTuple], Tuple[NamedTuple]], 
        target_MB: Union[int, float]=128.0, transformer: Optional[Union[DataTransformer, Callable]]=None
        ):
        """
        Args:
            config: List/Tuple of namedtuples containing configuration data for submission
            and record datasets, respectively. Must contain record_type, dataset_path, 
            and PyArrow schema.
            target_MB: Int or float value of target buffer size before *.parquet 
            conversion.
            transformer: Optional DataTransformer object to clean and preprocess a PyArrow
            table before writing to disk as *.parquet file. Must have either .transform method
            or implement __call__.
        
        Raises:
            ValueError: If input arguments are invalid.
            AttributeError: If a NamedTuple in config is missing required attributes.
        """
        logger.info('Initializing ParquetDataLoader with config: %s and target_MB: %s', config, target_MB)
        self._validate_init_args(config, target_MB, transformer)
        self.config = config
        self._transformer = transformer
        self.set_target_mb(target_MB)
        self._configure_loader() # Configure loader will catch invalid namedtuples and throw AttributeError
    
    def _validate_init_args(self, config, target_MB, transformer):
        """
        Abstraction of constructor argument validation checks.
        
        Args:
            config: Tuple containing configuration info.
            target_MB: Float/int of target buffer size in MB.
            transformer: DataTransformer object or Callable.
        
        Raises:
            ValueError: If input arguments are invalid.
            AttributeError: If a NamedTuple in config is missing required attributes.
        """
        # If list/tuple but != 2 elements, or if not list/tuple
        if (isinstance(config, (list, tuple)) and len(config) != 2) or not isinstance(config, (tuple, list)):
            raise ValueError(
                'Constructor expects a tuple or list of namedtuples containing\n'
                'record type ("submission", "comment"), dataset path, and PyArrow\n'
                'schema for submission and comment data, respectively.'
                )
        # Validate that target_MB is positive non-zero numeric value.
        if target_MB <= 0:
            raise ValueError('target_MB must be a positive non-zero int/float. Default = 32 MB.')
        if not isinstance(target_MB, (float, int)):
            raise TypeError(
                'Cannot assign non-numeric value to target_MB. Call constructor with\n'
                'a positive non-zero int/float.'
                )
        # Transformer must have at least a .transform() method or must be a 
        # callable (i.e. implements __call__()) that takes a PyArrow as input
        if transformer:
            has_transform = lambda obj: getattr(obj, 'transform', False)
            if not (callable(transformer) or has_transform(transformer)):
                raise ValueError(
                    'Transformer has no .transform method. Transformer can either be an object\n'
                    'implementing .transform or a Callable.'
                    )
    
    def _configure_loader(self):
        """
        Abstracts away the configuration of schemas, buffers, root paths, byte counters.
        
        Returns:
            self: Updated instance.
        """
        # Unpack schema from ParquetConfig namedtuples
        self._schemas = {ntuple.record_type : ntuple.schema for ntuple in self.config}
        # Store the dataset directory root paths per record type
        self._dataset_paths = {ntuple.record_type : ntuple.dataset_path 
                               for ntuple in self.config}
        # Set-up buffers per record type
        self._buffers = {
            record_type : {col : [] for col in self._schemas[record_type].names} 
            for record_type in self._schemas
            }
        # Initialize byte counter per record type
        self._buffer_sizes = {record_type : 0 for record_type in self._buffers}
        # Batch counter for filename
        self._batch_counters = {record_type : 0 for record_type in self._buffers}
        
        logger.debug('Loader schemas set to: %s', self._schemas)
        logger.debug('Dataset paths: %s', self._dataset_paths)
        return self
    
    def set_target_mb(self, target_MB: Union[int, float]):
        """
        Setter for target_MB. Updates target_MB and TARGET_BYTES attributes. Allows 
        for updating buffer size targets post-initialization.
        
        Args:
            target_MB: Target buffer size before *.parquet conversion.
        
        Returns:
            self: Updated instance.
        """
        self.target_MB = target_MB
        self._target_bytes = int(target_MB * 2**20)
        logger.info("Updating target buffer size to %d MB (%d bytes)", target_MB, self._target_bytes)
        return self
    
    def load(self, data_stream:Generator, partition_by_date: bool=False):
        """
        Streams data ("record type", record_dict) from an input generator, stores the 
        data into a dictionary buffer, and writes to disk when a target byte size or 
        when the function call has finished.
        
        Args:
            data_stream: Generator of submission and comment data. Use either 
            DataStreamer.stream() or DataStreamer.stream_search_results() for multi-
            subreddit multi-query or single subreddit query, respectively.
            partition_by_date: If True, scraped data will be stored in a subdir within dataset
            based on date when etl script was run.
        
        Notes:
            *.parquet files written to specified dataset directories in repo root.
            Default = UsedCarAnalytics/data/processed/**-dataset/**.parquet
        """ 
        logger.info('Starting to load data from stream...')
        
        if partition_by_date:
            # Add subdir to current dataset path to partition by date of scraping
            self._dataset_paths = {record_type: path / f'{dt.datetime.now():%Y-%m-%d}' 
                                   for record_type, path in self._dataset_paths.items()
                                   }
        
        try:
            # Stream the data, append to buffer, track buffer size, and when target buffer size
            # is met, write the record batch to disk and flush the buffer
            for record_type, record in data_stream:
                buffer = self._buffers[record_type]    
                
                for col in buffer:
                    buffer[col].append(record.get(col))
                
                # Update byte count with current record bytes
                record_bytes = len(json.dumps(record, separators=(",", ":")).encode("utf-8"))
                self._buffer_sizes[record_type] += record_bytes
                
                # Export parquet files to dataset directory and flush buffers and byte counters            
                if self._buffer_sizes[record_type] >= self._target_bytes:
                    logger.info('Target buffer size reached for %s. Writing batch #%d to disk.', record_type, self._batch_counters[record_type])
                    self._write(record_type, buffer)
                    # Increment batch counter for next batch file name
                    self._batch_counters[record_type] += 1
                    buffer, self._buffer_sizes[record_type] = self._flush(buffer)
                    logger.debug('Buffer flushed for %s after batch write.', record_type)
        
        except Exception as e:
            logger.error('Exception encountered during streaming: %s. Attempting to flush buffers to disk.', e, exc_info=True)
            # When an exception is encountered, write current buffers to disk
            for record_type, buffer in self._buffers.items():
                if any(container for container in buffer.values()):
                    logger.warning("Flushing %s buffer to disk after error.", record_type)
                    self._write(record_type, buffer, prefix='error')
                    buffer, self._buffer_sizes[record_type] = self._flush(buffer)
            logger.critical('ETL script terminating due to error.')
            raise RuntimeError(f'Exception {e} encountered.')
        
        else:
            logger.info('Finished streaming. Flushing any remaining buffers to disk.')
            # Final write for remaining data in both buffers after streaming data
            # Only write if there are remaining records to avoid null records in Parquet file
            for record_type, buffer in self._buffers.items():
                if any(container for container in buffer.values()):
                    logger.info("Flushing remaining %s buffer to disk.", record_type)
                    self._write(record_type, buffer, prefix='final')
                    buffer, self._buffer_sizes[record_type] = self._flush(buffer)
       
    def _write(self, record_type: str, buffer: Dict[str, List], prefix: Optional[str]=None):
        """
        Convert buffer to Pyarrow container, write to Parquet, then flush buffer and 
        byte count.
        
        Args:
            record_type: Either 'submission' or 'comment' to denote current buffer
            being written to disk.
            buffer: Dictionary of column name keys and list values for either submission
            or comment data.
            prefix (Optional): File name prefix. Useful for marking the final write
            before load call is terminated.
        
        Notes:
            Writes *.parquet files to **/*-dataset directory.
        """
        schema = self._schemas[record_type]
        # File name; Ex. submission-0001-20250626-201522 
        # or final-submission-0002-20250626-223000 if prefix='final'
        fname = (
            (f'{prefix}-' if prefix else '')
            + f'{record_type.lower()}-'
            + f'{self._batch_counters[record_type]:04d}-'
            + f'{dt.datetime.now():%Y%m%d-%H%M%S}.parquet'
            )
        # Absolute path; parquet.py config returns absolute paths as PosixPath
        fpath = self._dataset_paths[record_type] / fname
        # Ensure that the parent directories exist
        fpath.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Writing %s batch to file: %s", record_type, fpath)
        # Convert current buffer to Pyarrow Table and write Parquet files to target directory
        pa_table = pa.Table.from_pydict(buffer, schema=schema)
        # Transform batched data before writing if applicable; either via .transform() or __call__()
        if self._transformer:
            pa_table = self._transformer(pa_table)
        # Write to disk via GZIP for higher compression at the cost of write time, ideal
        # for long term static storage        
        logger.debug('Writing buffer for %s. Batch #%d.', record_type, self._batch_counters[record_type])
        pq.write_table(pa_table, where=fpath, compression='GZIP')
        logger.info('Write complete: %s', fpath)
    
    @staticmethod
    def _flush(buffer: Dict[str, List]):
        """Returns a tuple of empty buffer and byte count, in order, for an input buffer."""
        logger.debug('Flushing buffer and resetting byte count.')
        # Reconstruct buffer and return with empty values
        # Also return 0 for assignment to byte count
        return ({col : [] for col in buffer}, 0)
    
    def __repr__(self):
        param_dict = {'config':self.config, 'target_MB':self.target_MB}
        return f'ParquetDataLoader({", ".join([f'{k}={v}' for k, v in param_dict.items()])})'
    
    def __get__(self):
        return {
            'CONFIG' : self.config,
            'target_MB' : self.target_MB,
            '_TARGET_BYTES' : self._target_bytes,
            '_SCHEMAS' : self._schemas,
            '_ROOT_PATHS' : self._dataset_paths,
            '_buffers' : self._buffers,
            '_buffer_sizes' : self._buffer_sizes,
            '_batch_counters' : self._batch_counters
        }