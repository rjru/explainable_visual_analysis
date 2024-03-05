import pandas as pd
import json
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder

# Cargamos el archivo iris.csv
file_path = 'data\iris.csv'
iris_df = pd.read_csv(file_path)

# Separar el conjunto de datos en data y target
target = iris_df["variety"]  # La columna 'species' es nuestro objetivo
data = iris_df.drop("variety", axis=1)  # Eliminamos la columna 'species' para obtener las características
data_norm = normalize(data)
data_normalized = pd.DataFrame(data_norm, columns=data.columns)
l_enc = LabelEncoder()
iris_df['Category_numeric'] = l_enc.fit_transform(target)

data_vectors_norm = data_normalized.values.tolist()
dim_names = data_normalized.columns.tolist()
labels_encoded = iris_df['Category_numeric'].tolist()
label_items = iris_df['Category_numeric'].unique().tolist()
label_names = iris_df["variety"].unique().tolist()
# Creación del diccionario para el JSON


# generamos las proyecciones. 

json_data = {
    "dataVectors": data_vectors_norm,
    "dimNames": dim_names,
    "label": labels_encoded,
    "labelItems": label_items,
    "labelNames": label_names  # En este caso, labelNames y labelItems serán iguales
}

#print(json_data)
# Guardado del archivo JSON
json_file_path = 'data\iris_converted.json'
with open(json_file_path, 'w') as json_file:
    json.dump(json_data, json_file, indent=4)
