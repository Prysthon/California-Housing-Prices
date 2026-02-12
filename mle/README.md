# Housing ML (MLE version)

This folder contains an MLE-style rebuild of the California Housing project.

## Structure
- `src/`: core Python package
- `scripts/`: runnable pipelines
- `notebooks/`: EDA only
- `artifacts/`: trained models and metrics

## About California Housing Prices
1. longitude: A measure of how far west a house is; a higher value is farther west
2. latitude: A measure of how far north a house is; a higher value is farther north
3. housingMedianAge: Median age of a house within a block; a lower number is a newer building
4. totalRooms: Total number of rooms within a block
5. totalBedrooms: Total number of bedrooms within a block
6. population: Total number of people residing within a block
7. households: Total number of households, a group of people residing within a home unit, for a block
8. medianIncome: Median income for households within a block of houses (measured in tens of thousands of US Dollars)
9. medianHouseValue: **target** Median house value for households within a block (measured in US Dollars)
10. oceanProximity: Location of the house w.r.t ocean/sea

## Run
```bash
python scripts/train.py