/**
 * A k-means clustering algorithm implementation.
 *
 * Author  : Rohith Subramanyam
 * Student ID: 9071646443
 * wisc username: subramanyam2
 * CS login: rohithvsm
 * Class title: CS 540 Section 1: Introduction to Artificial Intelligence by Jerry Zhu
 * HW1
 * Date: Sept 26th 2014
 */

import java.util.*;

public class KMeans {

    private static double calculateDistortion(double[][] centroids
                                             ,double[][] instances
                                             ,KMeansResult result) {

        double distortion = 0;

        // centroids
        for (int i = 0; i < centroids.length; i++) {
            // instances
            for (int j = 0; j < result.clusterAssignment.length; j++) {
                // instances belonging to centroid i
                if (result.clusterAssignment[j] == i) {
                    // dimensions of j
                    for (int n = 0; n < instances[j].length; n++) {
                        distortion += Math.pow(instances[j][n] - centroids[i][n], 2);
                    }
                }
            }
        }

        return distortion;
    }

    private static void updateCentroids(double[][] centroids
                                       ,double[][] instances
                                       ,KMeansResult result) {

        double total;
        // store all instances of a centroid
        ArrayList<Integer> centroidInstances  = new ArrayList<Integer>();
        int num_instances;  // no. of instances in a cluster

        for (int i = 0; i < centroids.length; i++) {
            centroidInstances.clear();

            // add all instances belonging to cluster i to the arraylist
            for (int j = 0; j < result.clusterAssignment.length; j++) {
                if (result.clusterAssignment[j] == i) {
                    centroidInstances.add(j);
                }
            }
            num_instances = centroidInstances.size();

            // iterate over all the dimensions of centroid i and update it
            for (int n = 0; n < centroids[i].length; n++) {
                // Average each dimension of all instances belonging to centroid i
                total = 0;
                for (int centroidInstance : centroidInstances) {
                    total += instances[centroidInstance][n];
                }
                centroids[i][n] = total / num_instances;
            }
        }
    }

    /* 1. Search among all the instances for the instance x whose distance is farthest from its assigned centroid.
     * 2. Choose x’s position as the position of c, the orphaned centroid.
     */
    private static void assignAnInstanceToOrphan(int orphanCentroidIdx
                                                ,double[][] instances
                                                ,double[][] centroids
                                                ,KMeansResult result) {

        double maxDist  = -1;
        int instanceIdx = -1;
        double currentDist;
        int centroidIdx;  // centroid each instance belongs to

        // get the instance that's the farthest from its centroid
        for (int i = 0; i < instances.length; i++) {
            centroidIdx = result.clusterAssignment[i];  // get instance's centroid

            currentDist = calculateEucledianDistance(instances[i], centroids[centroidIdx]);
            if (currentDist > maxDist) {
                maxDist     = currentDist;
                instanceIdx = i;
            }
        }

        // assign it to the orphan centroid
        //result.clusterAssignment[instanceIdx] = orphanCentroidIdx;
        centroids[orphanCentroidIdx] = instances[instanceIdx];
    }

    private static int checkForOrphanCentroid(double[][] centroids, KMeansResult result) {

        int orphanCentroidIdx = -1;
        Set<Integer> centroidSet = new HashSet<Integer>();

        for (int centroidIdx : result.clusterAssignment) {
            centroidSet.add(centroidIdx);
        }

        for (int i = 0; i < centroids.length; i++) {
            if (!centroidSet.contains(new Integer(i))) {
                orphanCentroidIdx = i;
                break;
            }
        }

        return orphanCentroidIdx;
    }


    /**
     * Calculates the Euclidean distance between the given vectors.
     */
    private static double calculateEucledianDistance(double[] vector1, double[] vector2) {

        /*if (vector1.length != vector2.length) {
            throw new IllegalArgumentException("Vectors should be of equal length. Lengths are "
                            + vector1.length + " and " + vector2.length);
        }*/

        double dist = 0.0;

        for (int i = 0; i < vector1.length; i++) {
           dist += Math.pow((vector1[i] - vector2[i]), 2);
        }

        return Math.sqrt(dist);
    }

    public KMeansResult cluster(double[][] centroids, double[][] instances, double threshold) {
        /* ... YOUR CODE GOES HERE ... */

        int centroidIdx;  // index of centroid in centroids closest to an instance
        double minDist;   // distance of an instance from its closest centroid
        double currentDist;

        KMeansResult result      = new KMeansResult();
        result.clusterAssignment = new int[instances.length];
        double distortion;  // distortion in the current iteration
        // store the distortion in each iteration
        ArrayList<Double> distortionIters = new ArrayList<Double>();

        int iteration     = -1;  // to ensure a min of 2 iterations
        boolean converged = false;

        while (!converged) {

            // allocate instances to centroids
            // for each instance
            for (int i = 0; i < instances.length; i++) {
                // get the closest centroid
                centroidIdx = -1;
                minDist     = Double.MAX_VALUE;

                for (int j = 0; j < centroids.length; j++) {
                    currentDist = calculateEucledianDistance(instances[i], centroids[j]);
                    if (currentDist < minDist) {
                        minDist     = currentDist;
                        centroidIdx = j;
                    }
                }
                result.clusterAssignment[i] = centroidIdx;
            }

            // check for orphan centroids
            //boolean noOrphanCentroids = false;  // initial assumption

            // while there are orphan centroids
            //while(!noOrphanCentroids) {
            int orphanCentroidIdx;

            orphanCentroidIdx = checkForOrphanCentroid(centroids, result);
            if (orphanCentroidIdx > -1) {
                assignAnInstanceToOrphan(orphanCentroidIdx, instances, centroids, result);
                continue;  // reallocate all instances
            } /*else {
                noOrphanCentroids = true;
            }
            }*/

            iteration++;
            updateCentroids(centroids, instances, result);

            distortion = calculateDistortion(centroids, instances, result);
            distortionIters.add(distortion);

            if (   iteration > 0  // done with at least 2 iterations
                && Math.abs((  distortionIters.get(iteration)
                             - distortionIters.get(iteration - 1)
                            ) / distortionIters.get(iteration - 1)) < threshold) {
                converged = true;
            }
        }

        // set result
        result.centroids = centroids;

        result.distortionIterations = new double[distortionIters.size()];
        for (int i = 0; i < result.distortionIterations.length; i++) {
            result.distortionIterations[i] = distortionIters.get(i);
        }

        return result;
    }
}
