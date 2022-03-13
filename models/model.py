import pandas as pd
from sklearn.model_selection import ParameterGrid
from catboost import Pool, CatBoostClassifier
from tqdm import tqdm
from collections import defaultdict
import ml_metrics


class RecommendationModel:

    def x_y_split(self, data: pd.DataFrame):
        """

        Args:
            data: pd.DataFrame with user-item interactions with all features with target

        Returns:
            X: pd.DataFrame with user-item interactions with all features without target and useless features
            y: pd.DataFrame with target of user-item interactions

        """
        X = data.drop(['target', 'user_id',	'item_id'], axis=1)
        y = data['target']

        return X, y

    def fit(self, train_data: pd.DataFrame, val_data: pd.DataFrame):
        """

        Args:
            train_data: train pd.DataFrame with user-item interactions with all features with target
            val_data: val pd.DataFrame with user-item interactions with all features with target

        """
        X_train, y_train = self.x_y_split(train_data)
        X_val, y_val = self.x_y_split(val_data)

        # grid = {'learning_rate': [1], 'depth': [4, 5]}
        grid = {'learning_rate': [1], 'depth': [5]}
        train_dataset = Pool(data=X_train,
                             label=y_train,
                             cat_features=[X_train.shape[1] - 1]) # last featuer is categorial
        eval_dataset = Pool(data=X_val,
                            label=y_val,
                            cat_features=[X_val.shape[1] - 1]) # last featuer is categorial

        best_roc_auc = -1
        print('Training begins:')
        for params in ParameterGrid(grid):
            model = CatBoostClassifier(random_seed=7, eval_metric='AUC')
            model.set_params(**params)

            # Fit model
            model.fit(train_dataset,
                      use_best_model=True,
                      eval_set=eval_dataset)

            if model.get_best_score()['validation']['AUC'] > best_roc_auc:
                best_roc_auc = model.get_best_score()['validation']['AUC']
                best_params = model.get_params()
                best_iteration = model.get_best_iteration()
                best_model = model
            if model.get_best_score()['validation']['AUC'] < 0:
                raise Exception('smth went wrong witn model')
        print('Training finished')
        print('best_model params:', best_model.get_params())

        self.model = best_model

    def predict(self, test_user_item_interact: pd.DataFrame) -> pd.DataFrame:
            """

            Args:
                test_user_item_interact: test pd.DataFrame with user-item positive interactions with
                    user features with item features

            Returns:
                test_user_item_interact: test pd.DataFrame with user-item positive interactions with
                    user features with item features and with probabilities

            """
            X_test = test_user_item_interact.drop(['user_id', 'item_id'], axis=1)
            test_dataset = Pool(data=X_test,
                                cat_features=[X_test.shape[1] - 1])

            preds_proba = self.model.predict_proba(test_dataset)[:, 1]

            test_user_item_interact['probs'] = pd.Series(preds_proba, index=test_user_item_interact.index)

            return test_user_item_interact

    def get_topK_items_to_user(self, test_user_item_interact, item_features, top_k: int = 10) -> pd.DataFrame:
        """

        Args:
            test_user_item_interact: test pd.DataFrame with user-item positive interactions with
                user features with item features and with probabilities
            item_features: pd.DataFrame with item features
            top_k: int value for getting k most probably items descending for each user

        Returns:
            top_k_items_to_user: pd.DataFrame with k most probably items descending for each user

        """

        if top_k <= 0:
            raise ValueError(f'Expected k > 0. Got {top_k}.')

        tmp = []
        sorted_test_user_item_interact = test_user_item_interact.sort_values(by=['user_id', 'probs'])
        len_all_items = len(set(list(item_features['row'].unique())))

        for user_number in tqdm(range(len(test_user_item_interact.groupby('user_id')))):
            start_idx = (user_number + 1) * len_all_items - top_k
            end_idx = (user_number + 1) * len_all_items
            user_id = sorted_test_user_item_interact.iloc[start_idx:end_idx][['user_id', 'item_id']].values[:, 0]
            item_id = sorted_test_user_item_interact.iloc[start_idx:end_idx][['user_id', 'item_id']].values[:, 1]

            tmp.extend(tuple(zip(user_id, item_id)))

        top_k_items_to_user = pd.DataFrame(tmp, columns=['row', 'col'])
        top_k_items_to_user = top_k_items_to_user.merge(test_user_item_interact, left_on=['row', 'col'], right_on=['user_id', 'item_id'])
        top_k_items_to_user = top_k_items_to_user.sort_values(by=['user_id', 'probs'], ascending=False)

        return top_k_items_to_user

    def calculate_map_k(self, top_k_items_to_user, test_data, top_k: int = 10):
        """

        Args:
            top_k_items_to_user: pd.DataFrame with k most probably items descending for each user
            test_data: test pd.DataFrame with user-item interactions with all features with target
            top_k: int value for calculating map@k

        Returns:
            mapk: (numpy.float64) map@k metric

        """
        if top_k <= 0:
            raise ValueError(f'Expected k > 0. Got {top_k}.')

        d_pred = defaultdict(list)
        for row in top_k_items_to_user[['user_id', 'item_id']].values:
            d_pred[row[0]].append(row[1])

        d_test = defaultdict(list)
        for row in test_data[['user_id', 'item_id']].values:
            d_test[row[0]].append(row[1])

        mapk = ml_metrics.mapk(d_test.values(), d_pred.values(), top_k)
        return mapk
