from collections import Counter
import math

def knn(data, query, k, distance_fn, choice_fn):
    neighbor_distances_and_indices = []

    # For each example in data
    for index, example in enumerate(data):
        
        # Calculate the distance between the query example and 
        # the current example from thee data.
        distance = distance_fn(example[:-1], query)

        # Add the distance and the index of the example to an ordered collection  
        neighbor_distances_and_indices.append((distance, index))

    # sort the ordered collection of distances and indices from
    # smallest to largest (in ascending order) by the distances
    sorted_neighbor_distances_and_indices = sorted(neighbor_distances_and_indices)

    # pick the first K entries from the sorted collection
    k_nearest_distances_and_indices = sorted_neighbor_distances_and_indices[:k]

    # Get the labels of the selected K entries
    k_nearest_labels = [data[i][1] for distance, i in k_nearest_distances_and_indices]

    # If regression (choice_fn = mean), return the average of the K labels
    # If regression (choice_fn = mode), return the mode of the K labels 
    return k_nearest_distances_and_indices, choice_fn(k_nearest_labels)

def mean(labels):
    return sum(labels)/len(labels)

def mode(labels):
    return Counter(labels).most_common(1)[0][0]

def euclidean_distance(point1, point2):
    
    sum_squared_distance = 0

    for i in range (len(point1)):
        sum_squared_distance += math.pow((point1[i]-point2[i]), 2)
    
    return math.sqrt(sum_squared_distance)

def main():

    '''
    Regression Data
    Column 0: Height(inches)
    Column 1: Weight(pounds)
    '''
    reg_data = [
        [65.75, 112.99],
        [71.52, 136.49],
        [69.40, 153.03],
        [68.22, 142.34],
        [67.79, 144.30],
        [68.70, 123.30],
        [69.80, 141.49],
        [70.01, 136.46],
        [67.90, 112.37],
        [66.49, 127.45]
    ]

    reg_query = [60]

    reg_k_nearest_neighbors, reg_prediction = knn(
        reg_data, reg_query, k=3, distance_fn=euclidean_distance, choice_fn=mean )
    
    print('The predicted weight of student is: ', reg_prediction)

    '''
    Classification Data
    Column 0: Height(inches)
    Column 1: class
    '''
    clf_data = [
        [22, 1],
        [23, 1],
        [21, 1],
        [18, 1],
        [19, 1],
        [25, 0],
        [27, 0],
        [29, 0],
        [31, 0],
        [45, 0]
    ]

    clf_query = [33]

    clf_k_nearest_neighbors, clf_prediction = knn(
        clf_data, clf_query, k=3, distance_fn=euclidean_distance, choice_fn=mode )
    print('The predicted class of student is: ', clf_prediction)

if __name__ == '__main__':
    main()

def recommend_movies(movie_query, k_recommendations):
    
    raw_movies_data = []
    with open ('movies_recommendation_data.csv', 'r') as md:
        # discard the first line (heading)
        next(md)
    
        # read the data into memory
        for line in md.readlines():
            data_row = line.strip().split(',')
            raw_movies_data.append(data_row)

    # prepare the data for use in the knn algorithm by picking
    # the relevant columns and converting the numeric columns
    # to numbers since they were read in as strings
    movies_recommendation_data = []

    for row in raw_movies_data:
        data_row = list(map(float, row[2:]))
        movies_recommendation_data.append(data_row)
    
    # Use the KNN algorithm to get the 5 movies that are most similar to the post
    recommendation_indices, _ = knn(movies_recommendation_data, movie_query, k=k_recommendations,
                                    distance_fn=euclidean_distance, choice_fn=lambda x: None)
    
    movie_recommendations = []

    for _, index in recommendation_indices:
        movie_recommendations.append(raw_movies_data[index])

    return movie_recommendations

if __name__ == '__main__':
    the_post = [7.2, 1, 1, 0, 0, 0, 0, 1, 0] # feature vector for the post
    recommended_movies = recommend_movies(movie_query=the_post, k_recommendations=5)

    # print recommended movie titles
    for recommendation in recommended_movies:
        print(recommendation[1])

