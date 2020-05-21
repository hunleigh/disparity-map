# disparity-map
An algorithm to produce a disparity map from a pair of black and white stereo images. Matching is done on epipolar lines. Consists of a forward pass to create a cost matrix, and a backtrack to determine optimal solution. Dynamic Programming problem. Time complexity for matching is O(nm). CPU intensive.
