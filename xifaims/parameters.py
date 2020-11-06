xgb_small = {'n_estimators': [50, 150, 250, 25],
             'max_depth': [3, 6, 9, 12],
             'learning_rate': [0.001, 0.01, 0.1, 0.3],
             'subsample': [0.90],
             'reg_alpha': [None],
             'colsample_bytree': [0.7],
             'min_child_weight': [1, 3],
             'gamma': [0.0], 'nthread': [1], 'seed': [42]}

xgb_tiny = {'n_estimators': [10, 50, 100],
            'learning_rate': [0.001, 0.01, 0.1, 0.3],
            'nthread': [1], 'seed': [42]}

xgb_large = {'n_estimators': [30, 50, 100],
             'max_depth': [3, 5, 7, 9],
             'min_child_weight': [0.001, 0.1, 1],
             'learning_rate': [0.001, 0.01, 0.1, 0.3],
             'gamma': [0.0, 0.1, 0.2, 0.3],
             'reg_alpha': [1e-5, 1e-2, 0.1],
             'reg_lambda': [1e-5, 1e-2, 0.1],
             'nthread': [1], 'seed': [42]}