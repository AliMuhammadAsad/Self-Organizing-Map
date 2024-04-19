from som import *

def load_data(filename):
    print(f"Loading data from {filename}...")
    data = pd.read_csv(filename)
    data = data.groupby(['Country_Region'])[['Confirmed', 'Recovered', 'Deaths']].sum().reset_index()
    data_scaled = data.copy()
    population, faltu = [], []

    for index, row in data.iterrows():
        try:
            population.append(CountryInfo(row['Country_Region']).population())
        except:
            faltu.append(index)
    data_scaled = data_scaled.drop(faltu)
    data_scaled['Population'] = population
    data_scaled['Confirmed'] = data_scaled['Confirmed'] / data_scaled['Population']
    data_scaled['Recovered'] = data_scaled['Recovered'] / data_scaled['Population']
    data_scaled['Deaths'] = data_scaled['Deaths'] / data_scaled['Population']

    # print(f"Data loaded successfully!\nYour data:\n{data_scaled}")
    print(f"Normalizing the dataset...")
    # Normalizing the dataset using min-max scaler to scale the values between 0 and 1
    data_scaled['Confirmed'] = (data_scaled['Confirmed'] - data_scaled['Confirmed'].min()) / (data_scaled['Confirmed'].max() - data_scaled['Confirmed'].min())
    data_scaled['Recovered'] = (data_scaled['Recovered'] - data_scaled['Recovered'].min()) / (data_scaled['Recovered'].max() - data_scaled['Recovered'].min())
    data_scaled['Deaths'] = (data_scaled['Deaths'] - data_scaled['Deaths'].min()) / (data_scaled['Deaths'].max() - data_scaled['Deaths'].min())

    # Formatting the data into the fields we are concerned with, and transposing it for ease in computation for the SOM algorithm
    data_fields = data_scaled[['Confirmed', 'Recovered', 'Deaths']].values
    formatted_data = np.zeros((data_scaled.shape[0],3))
    for i in range(1, data_scaled.shape[0]):
        try:
            formatted_data[i][0] = data_fields[i][0]
            formatted_data[i][1] = data_fields[i][1]
            formatted_data[i][2] = data_fields[i][2]
        except KeyError:
            pass

    formatted_data = np.delete(formatted_data, 0, 0)
    formatted_data=formatted_data.transpose()
    # print(f"Data normalized successfully!\nYour Data: \n{data_scaled}\nFormatted Data: \n{formatted_data}")
    return data_scaled, formatted_data

if __name__ == "__main__":
    data_scaled, formatted_data = load_data('Q1_countrydata.csv')
    grid = (10, 10)                 # Grid Size, x and y dimensions
    alpha = 0.01                    # Learning rate
    radius = max(grid) // 2         # Radius of the neighbourhood
    epochs = 5000                   # Number of iterations

    print(f"Training the SOM with the following parameters:\nGrid Size: {grid}\nLearning Rate: {alpha}\nRadius: {radius}\nEpochs: {epochs}")
    som = SOM(grid, alpha, radius, epochs, data_scaled, formatted_data)
    som.colorMap(som.neural_network, 0)
    som.train()
    som.colorMap(som.neural_network, epochs)
    som.worldMap()