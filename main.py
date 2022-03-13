from dataloader.dataloader import GettingData
from dataloader.prepare_data import get_positive_and_negative_interactions
from models.model import RecommendationModel


def main():
    d = GettingData('./data/external')
    interactions, item_features, user_features = d.get_external_data('interactions.csv', ['item_asset.csv', 'item_price.csv', 'item_subclass.csv'],
                                                                     ['user_age.csv', 'user_region.csv'])

    train_data, val_data, test_data, test_user_item_interact = get_positive_and_negative_interactions(interactions,
                                                                                            user_features, item_features)
    model = RecommendationModel()
    model.fit(train_data, val_data)
    test_user_item_interact = model.predict(test_user_item_interact)

    top_k_items_to_user = model.get_topK_items_to_user(test_user_item_interact, item_features)
    mapk = model.calculate_map_k(top_k_items_to_user, test_data)
    print('Gradient Boosting map@10 =', mapk)


if __name__ == '__main__':
    main()
