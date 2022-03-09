import pandas as pd 
import numpy as np 
import json
import time
import os 
import sys



path=(os.getcwd()+'\data')

print(path)


#Transformation of the data was broken into 3 parts with the each function described as it was required to work with 3 steps.
#Through 3 essential functions, the data transformation was completed and saved as excel file. 
#First action was preprocessing of the data. The data type came as a .json file. After importing the data, pandas dataframe 
#was used for any data manipulation and extractions particularly for this task.



df=pd.read_json(path+'\supplier_car.json',lines=True)


#First thing is, importing data from json file. Using pandas library is very efficient way.By using json library,
#reading file method could have  also been used. 
#Data had some features that came straightforward but at the same time there were some other features that embeded into two columns.
#Additionally, each unique id represented each items (Cars) but number of data points for the id's were many.
#This is why grouping the id column was the first thing to combine each item into single row. 

def pre_processing(data):
    
    #ID columns was partitioned into groups. Afterwards all the objects and features extracted into a new dataframe.
    #In this first data frame there were two columns as ID for unique number and All Features which included all 
    # the information respectively for the id number 
    pre_process_df=pd.DataFrame(data.groupby(data['ID']),columns=['ID','All Features'])
    # MakeText column was extracted.
    pre_process_df['MakeText']=[pre_process_df.iloc[i,1].reset_index(drop=True)['MakeText'][0] 
                            for i in range(0,len(pre_process_df))]
    # TypeName column was extracted.
    pre_process_df['TypeName']=[pre_process_df.iloc[i,1].reset_index(drop=True)['TypeName'][0] 
                            for i in range(0,len(pre_process_df))]
    # TypeNameFull column was extracted.
    pre_process_df['TypeNameFull']=[pre_process_df.iloc[i,1].reset_index(drop=True)['TypeNameFull'][0] 
                            for i in range(0,len(pre_process_df))]
    # ModelText column was extracted.
    pre_process_df['ModelText']=[pre_process_df.iloc[i,1].reset_index(drop=True)['ModelText'][0] 
                            for i in range(0,len(pre_process_df))]
    # ModelTypeText column was extracted.
    pre_process_df['ModelTypeText']=[pre_process_df.iloc[i,1].reset_index(drop=True)['ModelTypeText'][0] 
                            for i in range(0,len(pre_process_df))]

    #In this loop, all Attribute names and Attribute Values were extracted from All Features column for each unique item.
    for ind, each in enumerate(pre_process_df['All Features']):
        for indx,name in enumerate(each['Attribute Names']):
            pre_process_df.loc[ind,name]= each['Attribute Values'].iloc[indx]

    pre_process_df.index=pre_process_df.ID  #ID number assigned as index to ease the data manipulation later      
    pre_process_df.drop(['All Features'],axis=1,inplace=True) #After mining all the valuable data point, All Feature was no needed so removed.
    
    return pre_process_df



#This function was created to normalize some attributes of the data before the integration part. 
def normalization(data):

    processed=data.copy()
    
    #ConditionTypeText ['Occasion', 'Oldtimer', 'Neu', 'Vorführmodell'] was replaced with English words
    ConditionTypeText_old=list(processed["ConditionTypeText"].unique())
    ConditionTypeText_new=['Used','Restored','New','Original Condition']
    processed["ConditionTypeText"]=processed["ConditionTypeText"].replace(ConditionTypeText_old,ConditionTypeText_new)
    
    
    #ConsumptionTotalText feature had different representation in the Target data. This why it is changed as l_km_consumption or other or null.
    for indx,each in enumerate(processed["ConsumptionTotalText"],1):
    
        try:
            if each.split()[1] =='l/100km' :
                processed.loc[indx,"ConsumptionTotalText"]="l_km_consumption"
            else:
                processed.loc[indx,"ConsumptionTotalText"]="other"
        except:
             processed.loc[indx,"ConsumptionTotalText"]="null"
                
    # BodyColorText had different values as mainly not English comparing the Target Data.
    #In the data source, number of unique colors was more than Target Data however, I sticed with the unique colors representations of Target Data.
    #Apart from defined color names in Target Data, all other colors was defined as 'other' for the normalization of the colors
    BodyColorText_old=list(processed["BodyColorText"].unique())
    BodyColorText_new=['Other', 'Other', 'Beige', 'Beige', 'Blue',
        'Blue', 'Other', 'Other', 'Brown', 'Brown',
       'Yellow', 'Yellow', 'Gold', 'Gold', 'Gray', 'Gray',
       'Green', 'Green', 'Orange', 'Orange', 'Red', 'Red',
       'Black', 'Black', 'Silver', 'Silver', 'Purple',
       'White', 'White']
    processed["BodyColorText"]=processed["BodyColorText"].replace(BodyColorText_old,BodyColorText_new)
                  
    # BodyTypeText ['Limousine','Kombi','Coupé','SUV / Geländewagen','Cabriolet','Wohnkabine','Kleinwagen',
    #'Kompaktvan / Minivan','Sattelschlepper','Pick-up',nan] were converted into English words and closest matchings were picked.
    # Nevertheless, some of them might not be the exact fit for the models.
    BodyTypeText_old=list(processed["BodyTypeText"].unique())
    BodyTypeText_new=['Saloon', 'Station Wagon', 'Coupé', 'SUV', 'Convertible / Roadster',
       'Other', 'Other', 'Other',
       'Other', 'Other', 'Other']

    processed["BodyTypeText"]=processed["BodyTypeText"].replace(BodyTypeText_old,BodyTypeText_new)
    
    # Type of the models were mainly cars if the van and derived vehicels are considered as cars (Could also be categorized differently) 
    # However there was only one moterbike wich can not be classified as car so it was altered as motorbike .
    processed['type']=['car' if processed['TypeNameFull'][i] !='HARLEY-DAVIDSON HPU Hurricane TC' else 'motorbike' 
                       for i in range(1,len(processed)+1)]
    
    return processed

    #In the integration part, normalized data was used and this is the last step before all data was transformed by implementing the functions.
def integration(normalized_data):
    
    output=normalized_data.copy()
    
    
    output['ModelText']=output['ModelText'].fillna('null')
    
    #In the datframe that created, there were many columns that they were not existing  in Target Data.
    #That is why some of the columns were removed from current data frame. 
    output=output.drop(['ID','TypeName','TypeNameFull','TransmissionTypeText','DriveTypeText','Doors',
                    'InteriorColorText','Properties','Hp','FuelTypeText','Seats',
                    'Ccm','ConsumptionRatingText','Co2EmissionText'],axis=1)
    # Some of the colum names were different ,in spite of same values, between created dataframe and the Target Data
    # For the smooth integration process, names that were different were changed according to Target Data column names.
    revised_column_names = {'MakeText': 'make','ModelText':'model',
        'ModelTypeText': 'model_variant',
        'Km': 'mileage','ConditionTypeText':'condition','City':'city','BodyTypeText':'carType',
                     'FirstRegYear': 'manufacture_year','BodyColorText':'color', 'FirstRegMonth':'manufacture_month',
                       'ConsumptionTotalText':'fuel_consumption_unit'}
 

    output.rename(columns=revised_column_names,
          inplace=True)
    
    # Mileage unit has no data however we know that based on the values we have, they were labelled as Km so kilometer string objects were generated.
    output['mileage_unit'] = "kilometer"
    
    #['Zuzwil', 'Sursee', 'Porrentruy', 'Safenwil', 'Basel',St. Gallen'] as all the cities in Switzerland CH country code was generated
    output["country"] = "CH"   
   
    output["zip"] = 'null'  # No zip data 
    
    output["price_on_request"] = "null" # No price_on_request data
    
    output["drive"] = "null" # No drive data
    
    output["currency"] = "null" # No currency data

    
    # For the integration, sequential order of the Target Data had to be matched, that is why it was re-ordered 
    output=output[['carType','condition','currency','drive','city','country','make','manufacture_year',
                 'mileage','mileage_unit','model','model_variant','price_on_request','type',
                  'zip','manufacture_month','fuel_consumption_unit']]
    
    output.reset_index(drop=True, inplace=True)
    
    return output

#start = time.time()

pre=pre_processing(df)
nor=normalization(pre)
out=integration(nor)

def combining(df):
    out.to_excel("output.xlsx",index=False)  

    cwd = os.path.abspath('') 
    files = os.listdir(cwd)  


    empty_df = pd.DataFrame()
    for file in files:
        if file.endswith('.xlsx'):
             df = empty_df.append(pd.read_excel(file), ignore_index=True) 
    return df.to_excel('transformed+_integrated.xlsx')



with pd.ExcelWriter("transformation_process.xlsx") as writer:
    pre.to_excel(writer, sheet_name='pre_processing', index = False)
    nor.to_excel(writer, sheet_name='normalization', index = False)
    out.to_excel(writer, sheet_name='target_output', index = False)

    
#end = time.time()
#print(end - start)

