DATA SOURCE - MOVIE SUCCESS PREDICTION

Link to the dataset:
https://drive.google.com/file/d/1WioKz2VmD1T0SFEJLyiaujQgoPIsdgz7/view?usp=sharing 

Description on data:
A movie’s success can be interpreted in various ways: number of awards, total revenue, and box office. But a basic guideline is that a movie is considered profitable if the revenue is twice the budget. So, success will be a boolean feature that is true if the movie revenue exceeds twice the budget. I will be attempting to predict this value using several classification algorithms: logistic regression, K-nearest neighbors, decision tree, and random forest.
The dataset consists of 375,377 movies dating from 1884 to 2018. It consists of the following features:
· id – unique movie ID for TMDB
· title
· original title – alternate title for foreign countries
· release date
· budget – rounded to nearest dollar
· revenue – rounded to nearest dollar
· popularity – value updated daily by TMDB. Considers page views, votes, activity, etc.
· runtime – rounded to nearest minute
· vote_average – average user ratings on a scale of 1-10
· vote_count – number of voters
· adult – boolean value. True if movie is a pornographic film
· status – state of movie production. i.e. released, in production, canceled
· genres
· production_companies
· production_countries
· certification_US – movie rating that determines suitability by viewer age
