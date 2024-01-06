from torch import nn

class Aggregator:

    def aggregate(self, weights: dict, freq: dict):
        aggr = {}

        for client, weight in weights.items():
            for key, value in weight.items():
                if aggr.get(key) is None:
                    aggr[key] = value * freq[client]
                else:
                    aggr[key] += value * freq[client]

        return aggr
        
    def aggregate_update(self, updates: dict, freq: dict):
        glob_update = {}
        
        for client, update in updates.items():
            for key, value in update.items():
                if glob_update.get(key) is None:
                    glob_update[key] = value * freq[client]
                else:
                    glob_update[key] += value * freq[client]
                    
        return glob_update
    
class SplitAggregator:

    def aggregate_update(self, updates: tuple[dict, dict], freq: dict):
        he_update, dp_update = updates
        glob_he_update, glob_dp_update = {}, {}

        for client, update in he_update.items():
            for key, value in update.items():
                if glob_he_update.get(key) is None:
                    glob_he_update[key] = value * freq[client]
                else:
                    glob_he_update[key] += value * freq[client]

        for client, update in dp_update.items():
            for key, value in update.items():
                if glob_dp_update.get(key) is None:
                    glob_dp_update[key] = value * freq[client]
                else:
                    glob_dp_update[key] += value * freq[client]
                    
        return glob_he_update, glob_dp_update