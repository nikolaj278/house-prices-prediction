

def sub_missing(df):
    # Identify columns that contain missing values
    cols_with_nulls = [c for c in df.columns if df[c].isnull().sum() > 0]
    
    # Replace missing values in categorical columns with 'missing'
    str_cols_with_nulls = df[cols_with_nulls].select_dtypes(include=['object', 'category']).columns
    df.fillna({column: 'missing' for column in str_cols_with_nulls}, inplace=True)
    
    # Replace missing values in numeric columns with mean (except GarageYrBlt)
    df.fillna({'LotFrontage': df['LotFrontage'].mean(),
               'MasVnrArea': df['MasVnrArea'].mean()}, inplace=True)
    
    # For GarageYrBlt, a null value indicates "no garage"
    # Create HasGarage binary feature: 1 if GarageYrBlt present, 0 if null
    df['HasGarage'] = df['GarageYrBlt'].notnull().astype(int)
    
    # Replace missing GarageYrBlt with 0 for rows with no garage
    df.fillna({'GarageYrBlt': 0}, inplace=True)
    
    return df