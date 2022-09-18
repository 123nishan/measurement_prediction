------Folder description-------------
    -dutch
        This folder contains csv files for demographic and measurement csv files
    -italy
        This folder contains csv files for demographic and measuremnet csv files

-------Pre processing of data-------------
    -This includes merge of Dutch and italy data, removing empty row, feature engineering (calculation for inseam height from measurement present in csv file)
    - split_data.py file contains code for pre-processing of data
        - to run use this command: python split_data.py "Gender"
            Here Gender can be male or female
            e.g. python split_data.py male