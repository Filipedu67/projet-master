supported_versions = ['1', '2']

# data version to use
# 1: original data
# 2: valeurs foncieres data
VERSION = 2

if VERSION == 1:
    COLUMN_TO_PREDICT = 'price'
elif VERSION == 2:
    COLUMN_TO_PREDICT = 'valeur_fonciere'
    #COLUMN_TO_PREDICT = 'Valeur fonciere'

SUPPORTED_CITIES = [
    'bordeaux',
    'lille',
    'lyon',
    'marseille',
    'montpellier',
    'nantes',
    'nice',
    'paris',
    'strasbourg',
    'toulouse'
]

# Define the columns you want to keep
# IMPORTANT: When you add new columns, remember to handle their value type (conversion to int, etc.)
COLUMNS_TO_KEEP = ['price', 'elevator', 'location.lat', 'location.lon', 'surface', 'bedroom', 'floor',
                   'furnished', 'room', 'propertyType', 'city.department.code']

COLUMNS_TO_KEEP_V2 = ['Valeur fonciere', 'No voie', 'B/T/Q', 'Type de voie',
                      'Code voie', 'Voie',
                      'Code postal', 'Code departement', 'Code commune', 'Commune',
                      '1er lot',
                      'Surface Carrez du 1er lot', '2eme lot', 'Surface Carrez du 2eme lot', '3eme lot',
                      'Surface Carrez du 3eme lot',
                      '4eme lot', 'Surface Carrez du 4eme lot', '5eme lot', 'Surface Carrez du 5eme lot',
                      'Nombre de lots',
                      'Type local', 'Code type local', 'Surface reelle bati',
                      'Nombre pieces principales',
                      'Surface terrain', 'No disposition', 'Nature mutation', 'Prefixe de section', 'Section', 'No plan',
                      'Nature culture', 'Nature culture speciale']
# Identifiant local

COLUMNS_TO_KEEP_V3 = ['valeur_fonciere',
                      'type_local', 'surface_reelle_bati', 'nombre_pieces_principales',
                      'surface_terrain', 'longitude', 'latitude']

# COLUMNS_TO_KEEP_V3 = ['valeur_fonciere', 'numero_disposition', 'adresse_numero',
#                       'adresse_code_voie', 'code_postal', 'code_commune', 'numero_volume', 'nombre_lots',
#                       'type_local', 'surface_reelle_bati', 'nombre_pieces_principales',
#                       'surface_terrain', 'longitude', 'latitude']

# 'nature_mutation', 'code_nature_culture'

# minimum and maximum price threshold
PRICE_THRESHOLD = [50000, 800000]

ADD_METRO_STATION = False

ENABLE_AI_DATA_SAVE = False

LIMIT_PRICE = True
