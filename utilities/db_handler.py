#***********************************
# SPDX-FileCopyrightText: 2009-2020 Vtenext S.r.l. <info@vtenext.com> and KLONDIKE S.r.l. <info@klondike.ai>
# SPDX-License-Identifier: AGPL-3.0-only
#***********************************

import mysql.connector as connector
from mysql.connector import Error
import pandas as pd
from tqdm import tqdm
import json
import pdb

IDS_COLUMN_SUFFIX = "_id" #needed in ml_methods too

def start_train(cron_id, start_date):
    
    data = json.load(open("utilities/connection_cron.json"))
    try:
        connection = connector.connect(host=data['host'], db=data['database'], user=data['user'], 
            password=data['psw'], port=data['port'])
        query = "UPDATE {} SET {} = 2, started = '{}' WHERE {} = {}".format(data['cron_table'], data['completed_name'], start_date, data['cron_id'], cron_id)
        print(query)
        cursor = connection.cursor()
        cursor.execute(query)
        connection.commit()

        print("train started at " + start_date + "\n")
        return 

    except Error as e:
        print("Error when connecting to CRON table\n")
        print(e)

def notify_training_completed(cron_id, additional_fields = []):
    
    data = json.load(open("utilities/connection_cron.json"))
    try:
        connection = connector.connect(host=data['host'], db=data['database'], user=data['user'], 
            password=data['psw'], port=data['port'])
        query = "UPDATE {} SET {} = 1 WHERE {} = {}".format(data['cron_table'], data['completed_name'], data['cron_id'], cron_id)
        print(query)
        cursor = connection.cursor()
        cursor.execute(query)
        connection.commit()

        for pair in additional_fields:
            query = "UPDATE {} SET {} = '{}' WHERE {} = {}".format(data['cron_table'], pair[0], pair[1], data['cron_id'], cron_id)
            print(query)
            cursor = connection.cursor()
            cursor.execute(query)
            connection.commit()

        print("CRON update complete")
        return 

    except Error as e:
        print("Error when connecting to CRON table\n")
        print(e)

def get_service_id(cron_id):

    data = json.load(open("utilities/connection_cron.json"))
    try:
        connection = connector.connect(host=data['host'], db=data['database'], user=data['user'], 
            password=data['psw'], port=data['port'])
        query = "SELECT {} FROM {} WHERE {} = {}".format(data['service_id'], data['cron_table'], data['cron_id'], cron_id)
        print(query)
        cursor = connection.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        return result[0][0]

    except Error as e:
        print("Error when connecting to CRON table\n")
        print(e)

def get_service_parameters(cron_id):
    
    service_id = get_service_id(cron_id)

    data = json.load(open("utilities/connection_service.json"))
    try:
        connection = connector.connect(host=data['host'], db=data['database'], user=data['user'], 
        password=data['psw'], port=data['port'])
        query = "SELECT {}, {}, {}, {}, {}, {} FROM {} WHERE {} = {}".format(
            data['table_name_field'], data['column_names_field'], data['target_name_field'], data['where_clause_field'], data['table_key_field'], 
            data['parameters_field'], data['service_table'], data['service_id'], service_id)
        print(query)
        cursor = connection.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        if result[0][3] is not None and ";" in result[0][3]:
            raise Exception('UNSAFE where_clause')
        if result[0][3] is None:
            where_clause = "" 
        else:
            where_clause = result[0][3]
        
        result_data  =    {
             "table_name": result[0][0],
             "column_names": result[0][1],
             "target_name": result[0][2],
             "table_key": result[0][4],
             "where_clause": where_clause,
             "service_id": service_id,
             "parameters": result[0][5]
        }    

        return result_data
     
    except Error as e:
        print("Error when connecting to SERVICE table\n")
        print(e)


def close_db_connection(connection):
    """
    Closes an open connetcion to the database.
    :param connection:
    """
    try:
        if connection.is_connected():
            connection.close()
    except Error as e:
        print("Error when closing the database")
        print(e)


def create_db_connection():
    """
    Creates a connection with mysql-connector package. Return the connector.
    :param host: string
    :param database: string, name of the databse
    :param user: string
    :param psw: string
    :param port: int
    :return: connection of type "mysql.connector"
    """
    data = json.load(open("utilities/connection.json"))
    try:
        connection = connector.connect(host=data['host'], db=data['database'], user=data['user'], 
            password=data['psw'], port=data['port'])
        return connection

    except Error as e:
        print("Error when connecting to database\n")
        print(e)


def get_db_training_data_OLD(column_list, table, table_key, target_col, where_clause):
    """
    Executes the query that fetches tha training data.
    :param column_list: list: columns to search in the db
    :param table: string: table to search into
    :param where_clause: string: optional where clause
    :return: DataFrame containing training data
    """

    query = "SELECT count(*) FROM {} WHERE coalesce({}) <> ''".format(table, target_col)
    if where_clause != '':
         query = query + ' AND ' + where_clause
    print(query)
    connection = create_db_connection()
    cursor = connection.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    close_db_connection(connection)
    print(str(result[0]) + " rows to process")
    
    query = "SELECT {},{} FROM {} WHERE coalesce({}) <> ''".format(table_key, ','.join(column_list), table, target_col)
    if where_clause != '':
         query = query + ' AND ' + where_clause
    print(query)
    data = pd.DataFrame(columns=column_list)

    connection = create_db_connection()
    try:
        if connection.is_connected():
            cursor = connection.cursor()
            cursor.execute(query)
            result = cursor.fetchall()

            # Optimization. Create tmp outside, add row, then delete row. In next iteration add row
            i = 0
            for row in result:
                l = list(row)
                l.pop(0)
                tmp_df = pd.DataFrame(l)
                tmp_df = tmp_df.transpose()
                tmp_df.columns = column_list
                data = data.append(tmp_df, ignore_index=True)
                i = i + 1
            print("{} rows".format(i))
            close_db_connection(connection)
            return data
        else:
            print("Connect to databse first")

    except Error as e:
        print("Error when executing the query to get data from db.\n\n")
        print(e)


def get_db_training_data(column_list, table, table_key, target_cols, where_clause, get_ids = False):
    """
    Executes the query that fetches tha training data.
    :param column_list: list: columns to search in the db
    :param table: string: table to search into
    :param target_cols: string or list of strings : classification targets
    :param where_clause: string: optional where clause
    :return: DataFrame containing training data
    """
    
    if type(target_cols) is list:
        query = "SELECT {},{} FROM {} WHERE"
        target_count = len(target_cols)
        for t in range(0,target_count):
            query = query + " coalesce({},'') <> ''"
            if t < target_count - 1:
                query = query + " AND"
        query = query.format(table_key, ','.join(column_list), table, *target_cols)
    else:
        query = "SELECT {},{} FROM {} WHERE coalesce({},'') <> ''".format(table_key, ','.join(column_list), table, target_cols)
    
    if where_clause != '':
         query = query + ' AND ' + where_clause
    print(query)
    
    connection = create_db_connection()
    try:
        if connection.is_connected():
            cursor = connection.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            
            df = pd.DataFrame.from_records(result, columns=[IDS_COLUMN_SUFFIX] + column_list) 
            if get_ids is False:
                df = df.drop(columns=[IDS_COLUMN_SUFFIX])  

            print("Total of " + str(len(df.index)) + " to be processed")
            
            close_db_connection(connection)
            
            df.fillna(" ",inplace=True)
            return df
        else:
            print("Connect to databse first")

    except Error as e:
        print("Error when executing the query to get data from db.\n\n")
        print(e)


def get_object_description(ticket_id, table_name, key_name, column_list):
    """
    Makes a query ad db to download the description of the associated ticket_id
    :param ticket_id:
    :return: data: pandas.DataFrame
    """
    print(column_list)
    query = "SELECT {} FROM {} WHERE {} = {}".format(','.join(column_list), table_name, key_name, ticket_id)
    print(query)
    result = None
    try:
        connection = create_db_connection()
        if connection.is_connected():
            cursor = connection.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
    except ConnectionError as e:
        result = None
        print("Error when trying to fetch ticket description: ", e)

    try:
        data = pd.DataFrame()
        for row in result:
            tmp_df = pd.DataFrame([list(row)])
            tmp_df.columns = column_list
            data = data.append(tmp_df, ignore_index=True)
        return data
    except ValueError as e:
        print("Error when creating variable from ticket description: \n\n", e)


def save_predictions_in_db(cron_id, ticket_id, prediction):
    """
    Executes the query that saves in the databse the predicted label corresponding to the text of ticket number
    "ticket_id".
    :param ticket_id: int, number of the ticket
    :param prediction: string: predicted label
    """
    data = json.load(open("utilities/connection_predictions.json"))
    try:
        connection = connector.connect(host=data['host'], db=data['database'], user=data['user'], 
            password=data['psw'], port=data['port'])
        query = "INSERT INTO {} ({},{},{}) VALUES (%s, %s, %s)".format(data['table'], data['table_record_column'], data['table_cron_column'], data['table_guessed_column'])
        cursor = connection.cursor()
        cursor.execute(query, (ticket_id, cron_id, str(prediction)))
        connection.commit()
        print("Prediction inserted successfully")
        return

    except Error as e:
        print("Error when executing the query to save predictions.")
        print(e)


def save_predictions_in_db_batch(cron_id, ids, predictions):
    """
    Executes the query that saves in the databse the predicted labels corresponding to the text of tickets with id in ids.
    :param ids: ticket ids list
    :param predictions: predictions list
    """
    data = json.load(open("utilities/connection_predictions.json"))
    try:
        connection = connector.connect(host=data['host'], db=data['database'], user=data['user'], 
            password=data['psw'], port=data['port'])
        query = "INSERT INTO {} ({},{},{}) VALUES (%s, %s, %s)".format(data['table'], data['table_record_column'], data['table_cron_column'], data['table_guessed_column'])
        
        values = list()
        for j in range(len(ids)):
            values.append((str(ids[j]),cron_id,str(predictions[j])))
        
        cursor = connection.cursor()
        cursor.executemany(query, values)
        connection.commit()
        print("Prediction batch inserted successfully")
        return

    except Error as e:
        print("Error when executing the query to save batch predictions.")
        print(e)
