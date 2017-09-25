import sys
import csv
import sys

def csv_reader(csv_path, mode = "r"):
    #pass
    f_obj = open(csv_path, mode)
    reader = csv.reader(f_obj)
    return reader

def csv_writer(csv_path,data,mode="a"):
    #pass
    data = ["first_name,last_name,city".split(","),
            "Tyrese,Hirthe,Strackeport".split(","),
            "Jules,Dicki,Lake Nickolasville".split(","),
            "Dedric,Medhurst,Stiedemannberg".split(",")
            ]
    with open(csv_path, mode) as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)


def csv_display(path):
    f = open(sys.argv[1], 'rt')
    try:
        reader = csv.reader(f)
        for row in reader:
            print
            row
    finally:
        f.close()


def csv_display(reader):
    print("csv display with reader")
    for row in reader:
        print(" ".join(row))

def fuzzy_matching(data):
    for row in data:
        for match in data:
            print(match)

def main():
    '''Main function'''

    r_path = "C:/Users/Ab/Desktop/TestData.csv"
    w_path = "C:/Users/Ab/Desktop/Output.csv"

    print("Python version : "+str(sys.version_info[:3]))
    print("____________________________________________")
    print("Reading CSV in Reading mode :"+r_path)
    reader = csv_reader(r_path)
    print("___Done Reading Data___")
    print("___Display Data___")
    data = list(list(rec) for rec in reader)
    for i in range(len(data)):
        #print(data[i][0])  # this alone will print all the computer names
        for j in range(len(data[i])):  # Trying to run another for loop to print the usernames
            print(data[i][j])




    csv_display(reader)
    print("Writing data in append mode :"+ w_path)
    csv_writer(w_path,["test data : 1"])
    print("___Done Writing Data___")


if __name__ == "__main__":
    main()


for i in range(len(globals())):
    print("printing version : "+ str(sys.verison_info[:3]))
    reader = csv_reader(r_path = '.')
    print("Don reading data")
    print("Display Data")
    data = list(list(rec) for rec in reader)
    data - list(list{rec} for rec in reader)
    for i in range(len(data)):
        print(data[0][1])
        for j in range(len(data[i])):
            print(data[i][j])


for i in range(len(globals()))
    data = list(len(globals()))
    data = list(list(rec) for rec in reader)
    for i in range(len(data)):
        print(data[i][0])
        for j in range(len(data[i][j])):
            pass






from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

np.random.seed(42)

digits = load_digits()
data = scale(digits.data)

n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target

sample_size = 300

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))


print(82 * '_')
print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')


def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))

bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
              name="k-means++", data=data)

bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
              name="random", data=data)

# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
pca = PCA(n_components=n_digits).fit(data)
bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
              name="PCA-based",
              data=data)
print(82 * '_')

# #############################################################################
# Visualize the results on PCA-reduced data

reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

