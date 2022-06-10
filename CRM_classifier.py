#***********************************
# SPDX-FileCopyrightText: 2009-2020 Vtenext S.r.l. <info@vtenext.com> and KLONDIKE S.r.l. <info@klondike.ai>
# SPDX-License-Identifier: AGPL-3.0-only
#***********************************

from AI.ml_methods import train_classifiers, predict_label
import getopt
import sys
import json
from utilities.db_handler import *
from datetime import datetime

def main():
    """
    :parameter: --train/--classify : whether to train classifiers or classify a new ticket

    """
    train = False
    predict = False
    ticket_id = None
    columns = None
    table = None
    target = None
    parameters = None

    try:
        options, remainder = getopt.getopt(sys.argv[1:], 'tci:d:s:w:z:b:e:f', ['train', 'classify', 'id=', 'columns=', 'table=', 'target=', 'from_db', 'cron_id=' , 'input_file='])

        db_parameters = False
        input_file = None

        for opt, arg in options:
            if opt in ('-t', '--train'):
                train = True
            elif opt in ('-c', '--classify'):
                predict = True
            elif opt in ('-i', '--id'):
                ticket_id = arg
            elif opt in ('-d', '--columns'):
                # NOTA BENE: I NOMI DELLE COLONNE DEVONO ESSERE NEL SEGUENTE FORMATO: "prima, seconda, terza"
                columns = arg
                print("columns: " + columns)
            elif opt in ('-s', '--table'):
                table = arg
                print("table: " + table)
            elif opt in ('-z', '--target'):
                target = arg
            elif opt in ('-b', '--from_db'):
                db_parameters = True
            elif opt in ('-e', '--cron_id'):
                cronid = arg
            elif opt in ('-f', '--input_file'):
                input_file = arg

        if train:
            where_clause = ""
            if db_parameters:
                assert cronid is not None  
                parameters_dict = get_service_parameters(cronid)
                table   = parameters_dict['table_name']
                table_key = parameters_dict['table_key']
                columns =parameters_dict['column_names']
                target  =parameters_dict['target_name']
                where_clause = parameters_dict['where_clause']
                serviceid = parameters_dict['service_id']
                parameters = parameters_dict['parameters']

            assert table is not None
            assert columns is not None
            assert target is not None
            
            columns = columns.split(",")
            columns.append(target)
            print("whole columns: " + ','.join(columns) )
            start_train(cronid, datetime.today().strftime("%Y-%m-%d %H:%M:%S"))
            report = train_classifiers(columns, table, table_key, target, where_clause, str(cronid) + "_" + str(serviceid), parameters)
            if db_parameters:
                notify_training_completed(cronid, [["training_result",json.dumps(report)],["ended",datetime.today().strftime("%Y-%m-%d %H:%M:%S")]])

        elif predict:

            model_prefix = ""

            if db_parameters:
               assert cronid is not None
               parameters_dict = get_service_parameters(cronid)
               serviceid = parameters_dict['service_id']
               model_prefix = str(cronid) + "_" + str(serviceid)
               parameters = parameters_dict['parameters']
               data_columns = parameters_dict['column_names']

            assert table is not None
            assert target is not None

            if input_file is None:  
                assert ticket_id is not None
                ticket_id = int(ticket_id)
                
            predict_label(cronid, ticket_id, table, target, data_columns, model_prefix, parameters, inputfile = input_file)

        else:
            raise ValueError("Error when parsing input parameters. No routine was set to True, so no training or "
                             "classification will run")

    except getopt.GetoptError as err:
        print(err)  # will print something like "option -a not recognized"
        sys.exit(2)


if __name__ == '__main__':
    main()
