from __future__ import division
import geopandas as gpd
import pandas as pd
import math, numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from matplotlib import cm as cm
from countryinfo import CountryInfo
from descartes import PolygonPatch

class SOM:
    def __init__(self, grid, learning_rate, sigma, epochs, data_scaled, formatted_data) -> None:
        self.grid = grid
        self.x = grid[0]; self.y = grid[1]
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.epochs = epochs
        self.data_scaled = data_scaled
        self.formatted_data = formatted_data
        self.features = formatted_data.shape[0]
        self.data_points = formatted_data.shape[1]
        self.lambda_tc = epochs / math.log(sigma)
        self.neural_network = np.random.rand(self.x, self.y, self.features)
        self.updated_weight_vectors = []

        # Color Schemes
        self.cmap = plt.cm.coolwarm                # take mean times 2
        # self.cmap = plt.cm.plasma_r                # take mean times 2
        # self.cmap = plt.cm.viridis                 # take mean times 2
        # self.cmap = plt.cm.rainbow_r               # separate values
        # self.cmap = plt.cm.seismic
        # self.cmap = plt.cm.autumn

    def updateRadius(self, epoch):
        return self.sigma * math.exp(-epoch / self.lambda_tc)

    def updateLearningRate(self, epoch):
        return self.learning_rate * math.exp(-epoch / self.lambda_tc)

    def neighborsInfluence(self, radius, distance):
        return math.exp(-distance / (2 * (radius ** 2)))

    def getEuclideanDistance(self, input_vector, nn, i, j):
        return math.sqrt((nn[i][j][0] - input_vector[0])**2 + (nn[i][j][1] - input_vector[1])**2 + (nn[i][j][2] - input_vector[2])**2)

    def findBestMatchingUnit(self, input_vector, nn):
        mini = float('inf')
        for i in range(self.x):
            for j in range(self.y):
                dist = self.getEuclideanDistance(input_vector, nn, i, j)
                if dist < mini:
                    mini = dist
                    bmu_idx = (i, j)
        bmu = nn[bmu_idx[0], bmu_idx[1]]
        return (bmu, bmu_idx)

    def colorMap(self, nn, epoch):
        print(f"Coloring the SOM Grid after {epoch} Epoch(s)...")
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ax.set_xlim((0, self.x + 1))
        ax.set_ylim((0, self.y + 1))
        ax.set_title('SOM Grid Coloring after %d Epoch(s)' % epoch)

        for i in range(self.x):
            for j in range(self.y):
                # Compute a single color value from the neuron's weights
                # color_value = np.mean(nn[i, j, :])
                # color_value = np.mean(nn[i, j, :]) * 2
                color_value = nn[i, j, 0] + nn[i, j, 1] + nn[i, j, 2]
                ax.add_patch(patches.Rectangle((i+0.5, j+0.5), 1, 1,
                                            facecolor=self.cmap(color_value),
                                            edgecolor='none'))
        plt.show()

    def train(self):
        for epoch in range(self.epochs):
            index = np.random.choice(self.data_points)
            input_vector = self.formatted_data[:, index].reshape(self.features, 1)
            bmu, bmu_idx = self.findBestMatchingUnit(input_vector, self.neural_network)
            r = self.updateRadius(epoch)
            lr = self.updateLearningRate(epoch)

            for i in range(self.x):
                for j in range(self.y):
                    curr_weight = self.neural_network[i, j, :].reshape(self.features, 1)
                    dist_to_bmu = np.linalg.norm([i - bmu_idx[0], j - bmu_idx[1]])
                    if dist_to_bmu <= r:
                        influence = self.neighborsInfluence(r, dist_to_bmu)
                        new_weight = curr_weight + (lr * influence * (input_vector - curr_weight))
                        if dist_to_bmu == 0:
                            tmp = []
                            tmp.append(new_weight.reshape(1, 3)[0])
                            tmp.append(index)
                            self.updated_weight_vectors.append(tmp)
                        self.neural_network[i, j, :] = new_weight.reshape(1, 3)
                        
            print(f"Epoch: {epoch}/{self.epochs}, Radius: {r}, Learning Rate: {lr}", end="\r")
    
    def worldMap(self):
        print("Plotting the World Map with the SOM Grid...")
        country_names = []
        for line in range(len(self.data_scaled)):
            try:
                if(self.data_scaled['Country_Region'][line] not in country_names):
                    country_names.append(self.data_scaled['Country_Region'][line])
            except KeyError:
                pass

        def least_dist(l1,nn):
            min_dist = float('inf')
            for i in nn:
                for j in i:
                    distance = math.sqrt((j[0] - l1[0])**2 + (j[1] - l1[1])**2 + (j[2] - l1[2])**2)
                    if distance < min_dist:
                        min_dist = distance
                        selected = j
            return selected

        def one_instance(weight_vectors):
            instance = []
            country=[]
            for vector in weight_vectors:
                if vector[1] not in country:
                    country.append(vector[1])
                    instance.append(vector)
            return instance

        o = one_instance(self.updated_weight_vectors)
        countries=[]
        country_names = []
        for i in o:
            best_weight=least_dist(i[0],self.neural_network)
            # c = self.cmap(np.mean(best_weight))
            # c = self.cmap(np.mean(best_weight) * 2)
            c = self.cmap((best_weight[0]+best_weight[1]+best_weight[2]))
            countries.append([i[1],c])

        countries_colors=[]

        for k in countries:
            j=k[0]
            news=[]
            try:
                news.append(CountryInfo(self.data_scaled['Country_Region'][j]).iso(3))
            except KeyError:
                pass
            news.append(k[1])
            countries_colors.append(news)

        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        ax2 = world.plot(figsize=(8,8), edgecolor=u'gray', color=u"white")

        for count in countries_colors:
            if (len(count)==1):
                continue
            else:
                # plot a country on the provided axes
                nami = world[world.iso_a3 == count[0]]
                namigm = nami.__geo_interface__['features']
                if namigm != []:
                    namig0 = {'type': namigm[0]['geometry']['type'], \
                            'coordinates': namigm[0]['geometry']['coordinates']}
                    ax2.add_patch(PolygonPatch( namig0, fc=count[1], ec="black", alpha=0.85, zorder=2 ))

        # the place to plot additional vector data (points, lines)
        plt.ylabel('Latitude')
        plt.xlabel('Longitude')

        plt.show()