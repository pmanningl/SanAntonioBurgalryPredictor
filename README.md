# SanAntonioBurgalryPredictor

Burglaries can have detrimental effects on housing prices, community, and an individual's 
feeling of safety.

Knowing your level of vulnurability can influence your home defence choices, or even your 
choice in housing location.

The following model makes use of burglary data for San Antonio TX 
from 05/2024-08/2024, and provides a UI for predicting the most vulnurable time for 
burglaries at a particular pair of coordinates.

The model is a basic neural network with 1000 hidden layers, and achieved a 
Mean Squared Error (MSE) of 0.4878237 after training on the set of 500 data
points (split 80:20 test:train)

## Future Improvements

- Collection of more datapoints from source (https://communitycrimemap.com/)
- Inclusion of monthly income as a feature based on census blocks
- DBSCAN clustering of predicted results

## Files

- MapDataTransformer.py --> transform source json data to .csv
- Models.py --> contains neural network model and various models for testing purposes
- **Primary File**    Mapper.py -->  uses pickled model and tkintermapview to generate interractive map (right click to select prediction coordinates) 
- Data: map_data.json, transformed_data.csv, mlp_model.pkl

## Instructions For Use
1. Run the Mapper.py to explore the map (Right click to set coordinates)
2. **Optional** tune parameters and re-run the Models.py to edit the model