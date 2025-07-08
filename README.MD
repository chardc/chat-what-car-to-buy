# Used Car Analysis from Reddit Discussions (Ongoing Project)

__Goal:__ Determine ideal used car models for my personal profile by leveraging crowd knowlege.

__Objectives:__
1. Establish access to Reddit API either via PRAW or Requests
2. Extract and parse relevant submissions and respective top-level comments
3. Explore and analyze textual data
4. Build a recommender that takes into account my personal profile and returns a comprehensive discussion on the ideal used vehicle model/s for me.

__Motivation:__ As a first-time car buyer, it's easy to get lost in the sea of options in the used car market. The plethora of options for brands, models, model years and trims, and whatnot, can make it daunting to do your own research. Personally, I find myself gravitating to online forums when doing my research on used cars as I believe in the concept of "Wisdom of the crowd", wherein majority sentiment may be indicative of some measurable and verifiable phenomena. 

In the case of used cars, if a significant number of people praise a specific Japanese-manufactured vehicle model while citing its faults, an algorithm that sufficiently captures this data may be able to present a high confidence level for the said model while presenting an overview of common issues to be aware of. In doing so, the work involved in research is cut in half, and prospective buyers can trim their options down to a few models that fit their criteria, while being made aware of the salient points that require extensive due diligence (e.g. common issues, upkeep, regulations) that only humans can do.

## How to run this program:
1. Configure your the .env file under the config directory with your client id and client secret keys from Reddit API. Also include your username, password, and agent in the .env file.
2. [OPTIONAL] Run the pipeline.py script to extract data from Reddit and update the datasets in the data directory. Otherwise, you can skip this step and utilize the pre-scraped data under the data directory.
3. [OPTIONAL] Check the notebooks to understand what each script does and the documentation for each function.
4. __TO BE CONTINUED__

## To be implemented:
1. Analysis on scraped text data
2. Recommender with text summarization for user-prompted queries (Ex. What's the best used sedan under $10,000)
3. UI for user prompting and displaying results
