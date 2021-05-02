Using the movies dataset, we build a recommender system:
- based on videos popularity using IMDB weighted rating formula
- based on the similarity in the overview of videos

 Dataset [Here](https://www.kaggle.com/rounakbanik/the-movies-dataset/data)

 ## Results:
 ### 1-

 | title  | vote_count  | vote_average | score |
| :------------: |:---------------:|:---------------:|:---------------:|
| The Shawshank Redemption     | 8358.0 | 8.5 | 8.445869 |
| The Godfather	| 6024.0 | 8.5 | 8.425439 |
| Dilwale Dulhania Le Jayenge | 661.0	 | 9.1 | 8.421453 |
| .......................     | .....| ..... | .....|

### 2-

```python
get_recommendations('Father of the Bride Part II')
```
| title  | 
| :------------:|
| Father of the Bride |
| Kuffs 6024.0 |
| North to Alaska|
| Wendigo |
| The Magic of Méliès |
| .......................|

[x] We'll improve again the second part to take account others features like the director, actors, keywords and the movie genres.

## Results
```python
get_recommendations('Father of the Bride Part II', cosine_similarity)
```
| title  | 
| :------------:|
| Baby Boom |
| Father of the Bride |
| ¡Three Amigos! |
| Hanging Up |
| Das merkwürdige Verhalten geschlechtsreifer Gr... |
| .......................|
