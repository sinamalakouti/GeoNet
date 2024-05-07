import os
import torch.utils.data as data
from PIL import Image
import json
from typing import List, Dict, Any

_VALID_DOMAIN = ["usa", "asia"]
_VALID_SPLIT = ["train", "test"]
_DEFAULT_STR = "NULL"

def default_loader(path):
    return Image.open(path).convert('RGB')

def default_ann(imid):
    return {
        "image_id":imid,
        "category": int(0),
        "class_name": _DEFAULT_STR
    }

def default_meta(imid):
    return {
        "image_id":imid,
        "caption": _DEFAULT_STR,
        "tags"       : _DEFAULT_STR,
        "url"        : _DEFAULT_STR
    }

def default_loc(imid):
    return {
        "image_id": imid,
        "countryRegionIso2": _DEFAULT_STR,
        "continentRegion"  : _DEFAULT_STR,
        "latitude"         : 0.0,
        "longitude"        : 0.0
    }


class ImageJSONLoader(data.Dataset):

    def __init__(self, root_dir, json_path, domain="usa", split="train", transform=None, target_transform=None,
                 loader=default_loader, return_ann=True, return_loc=False, return_meta=False, 
                 _loc_keys=None, _meta_keys=None):

        if _loc_keys is not None:
            assert isinstance(_loc_keys, list), "loc keys has to be a list."
            assert return_loc
            _loc_keys += list(default_loc(0).keys())
        else:
            _loc_keys = list(default_loc(0).keys())
        _loc_keys = list(set(_loc_keys))
        
        if _meta_keys is not None:
            assert isinstance(_meta_keys, list), "meta keys has to be a list."
            assert return_meta
            _meta_keys += list(default_meta(0).keys())
        else:
            _meta_keys = list(default_meta(0).keys())
        _meta_keys = list(set(_meta_keys))

        if not isinstance(domain, list):
            domain = [domain]
        if not isinstance(split, list):
            split = [split]

        assert all([d in _VALID_DOMAIN for d in domain]), "Invalid Domain".format(domain)
        assert all([s in _VALID_SPLIT for s in split]), "split has to be {}. {} not recognized".format("|".join(_VALID_SPLIT), split)
        self.top_150_asia = {'bazaar': 52, 'rose': 36, 'ox': 34, 'assembly_hall': 30, 'brain_coral': 30, 'bulbul': 30,
                             'cattleya': 30, 'cheetah': 30, 'church': 30, 'cock': 30, 'cormorant': 30, 'cosmos': 30,
                             'cow_parsnip': 30, 'dendrobium': 30, 'dome': 30, 'fiddler_crab': 30, 'flatworm': 30,
                             'gazelle': 30, 'gloriosa': 30, 'goby': 30, 'hippeastrum': 30, 'hornbill': 30,
                             'indian_elephant': 30, 'kimono': 30, 'lion': 30, 'lionfish': 30, 'macaque': 30,
                             'manhole_cover': 30, 'oncidium': 30, 'ostrich': 30, 'resort_area': 30, 'scorpionfish': 30,
                             'sea_anemone': 30, 'sea_cucumber': 30, 'sea_hare': 30, 'sea_pen': 30, 'sea_slug': 30,
                             'shoebill': 30, 'stupa': 30, 'tricycle': 30, 'trolleybus': 30, 'vending_machine': 30,
                             'wrasse': 30, 'airliner': 29, 'arabian_camel': 29, 'barn_swallow': 29, 'billboard': 29,
                             'brittle_star': 29, 'capybara': 29, 'colobus': 29, 'common_wood_sorrel': 29, 'hyena': 29,
                             'indian_rhinoceros': 29, 'jetliner': 29, 'lesser_panda': 29, 'mandarin_duck': 29,
                             'marabou': 29, 'osprey': 29, 'pasta': 29, 'restaurant': 29, 'sashimi': 29, 'streetcar': 29,
                             'triggerfish': 29, 'whale_shark': 29, 'zebra': 29, 'air_terminal': 28, 'batik': 28,
                             'boxfish': 28, 'bustard': 28, 'cattle_egret': 28, 'coral_reef': 28, 'cowrie': 28,
                             'curry': 28, 'giant_clam': 28, 'giant_panda': 28, 'hippopotamus': 28, 'impala_lily': 28,
                             'komodo_dragon': 28, 'leopard': 28, 'litter': 28, 'mantis_shrimp': 28,
                             'mountain_ebony': 28, 'oil_tanker': 28, 'pavilion': 28, 'rock_dove': 28, 'shrike': 28,
                             'starfish': 28, 'sunflower': 28, 'tiger': 28, 'walking_stick': 28, 'water_buffalo': 28,
                             'african_elephant': 27, 'apartment_building': 27, 'banquet': 27, 'electric_locomotive': 27,
                             'food_court': 27, 'frangipani': 27, 'kite': 27, 'mangrove': 27, 'peacock': 27,
                             'pedestrian_crossing': 27, 'scraper': 27, 'seashore': 27, 'sea_urchin': 27, 'wheatear': 27,
                             'black_rhinoceros': 26, 'breakwater': 26, 'dhole': 26, 'flower_cluster': 26, 'hyrax': 26,
                             'langur': 26, 'mausoleum': 26, 'puffer': 26, 'waterbuck': 26, 'yurt': 26, 'baboon': 25,
                             'bakery': 25, 'cape_buffalo': 25, 'cocoa': 25, 'florist': 25, 'flowerbed': 25,
                             'freight_car': 25, 'goldfish': 25, 'mongoose': 25, 'orangutan': 25, 'swamp': 25,
                             'verbena': 25, 'airframe': 24, 'ashcan': 24, 'cab': 24, 'dust_storm': 24, 'ginkgo': 24,
                             'headscarf': 24, 'lounge': 24, 'parrotfish': 24, 'passenger_car': 24, 'volcano': 24,
                             'wild_boar': 24, 'airfield': 23, 'bannister': 23, 'beach': 23, 'chimpanzee': 23,
                             'cuttlefish': 23, 'dune': 23, 'eland': 23, 'footbridge': 23, 'ladybug': 23,
                             'moth_orchid': 23, 'mountainside': 23, 'palace': 23}
        self.root_dir = root_dir

        keytag = []
        for d in domain:
            for s in split:
                keytag.append("{}_{}".format(d, s))
        json_data = json.load(open(json_path))
        # json_data = self.filter_all_data_based_on_categories(json_data, self.top_150_asia)
        print("number of objects is ", len(json_data['categories']))
        self.return_ann = return_ann
        self.return_loc = return_loc
        self.return_meta = return_meta

        self.info = json_data['info']
        self.category_mapping = json_data['categories']

        self.classname_to_id = {c["category_name"]:int(c["category_id"]) for c in self.category_mapping}
        self.id_to_classname = {v:k for k,v in self.classname_to_id.items()}
        
        imdata = [json_data[kt] for kt in keytag]
        id_to_im = {im["id"]:im for imd in imdata for im in imd["images"]}

        id_to_ann = {image_id:default_ann(image_id) for image_id in id_to_im.keys()}
        id_to_loc = {image_id:default_loc(image_id) for image_id in id_to_im.keys()}
        id_to_meta = {image_id:default_meta(image_id) for image_id in id_to_im.keys()}

        if return_ann:
            id_to_ann = {ann["image_id"]:ann for imd in imdata for ann in imd["annotations"]}
            assert len(id_to_ann) >= len(id_to_im), "Annotations Missing"

        if return_loc:
            id_to_loc = {loc["image_id"]:loc for imd in imdata for loc in imd["locations"]}
            assert len(id_to_loc) >= len(id_to_im), "Locations Missing"

        if return_meta:
            id_to_meta = {meta["image_id"]:meta for imd in imdata for meta in imd["metadata"]}
            assert len(id_to_meta) >= len(id_to_im), "Metadata Missing"

        ## combine image, annotation and locations
        self.geodata = []
        for imid in id_to_im.keys():

            self.geodata.append((
                imid,
                id_to_im[imid]["filename"],
                id_to_ann[imid]["category"],
                {k:id_to_loc[imid].get(k,"NULL") for k in _loc_keys},
                {k:id_to_meta[imid][k] for k in _meta_keys}
            ))

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.labels = [ann for _, _, ann, _, _ in self.geodata]
        print("hi")

    def filter_all_data_based_on_categories(self, all_data, categories_names):
        from copy import copy, deepcopy

        filtered_data = dict()
        categories_names = set(categories_names)
        print(f"Filtering the json_data to only include {len(categories_names)} most freq objects from asia test-set")
        #   Step 1 set the info
        filtered_data['info'] = f'Metadata for images in GeoImnet. Collected by Tarun Kalluri. 03/23. Filtered by Sina to include {len(categories_names)} most frequent categories based on the test set.'
        # Step 2 set the category list
        filtered_data['categories'] = []
        cat_counter = 0
        old_to_new_cat_id_mapping = {}
        for cat_item in all_data['categories']:
            if cat_item['category_name'] in categories_names:
                new_cat_item = copy(cat_item)
                old_to_new_cat_id_mapping[cat_item['category_id']] = cat_counter
                new_cat_item['category_id'] = str(cat_counter)
                filtered_data['categories'].append(new_cat_item)
                cat_counter += 1
        old_categories_ids = set(old_to_new_cat_id_mapping.keys())
    #     Step 3 : fix the usa_train, usa_test, asia_train, asia_test
        splits = ['usa_train', 'usa_test', 'asia_train', 'asia_test']
        for split in splits:
            filtered_data[split] = {} #fields: images, locations, metadata, annotations
            filtered_data[split]['images'] = []
            filtered_data[split]['annotations'] = []
            filtered_data[split]['metadata'] = []
            filtered_data[split]['locations'] = []
            for i in range(len(all_data[split]['images'])):
                image = copy(all_data[split]['images'][i])
                ann = copy(all_data[split]['annotations'][i])
                item_metadata = copy( all_data['usa_train']['metadata'][i])
                loc = copy(all_data[split]['locations'][i])
                if str(ann['category']) in old_categories_ids:
                    ann['category'] = int(old_to_new_cat_id_mapping[str(ann['category'])])
                    filtered_data[split]['annotations'].append(ann)
                    filtered_data[split]['images'].append(image)
                    filtered_data[split]['metadata'].append(item_metadata)
                    filtered_data[split]['locations'].append(loc)
        return filtered_data



    def _getitem_single(self, index):
        imid, impath, target, location, metadata = self.geodata[index]
        location.pop("image_id", None)
        metadata.pop("image_id", None)

        impath = os.path.join(self.root_dir, impath)
        img = self.loader(impath)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return_obj = [imid, img]

        if self.return_ann:
            return_obj.append(target)

        if self.return_loc:
            return_obj.append(location)

        if self.return_meta:
            return_obj.append(metadata)

        return_obj.append(index)

        return return_obj

    def __getitem__(self, index):
        # print("getitem  ",index)
        if isinstance(index, list) and len(index) > 0:
            batch = []
            for ind in index:
                res = self._getitem_single(ind)
                batch.append(res)
        else:
            batch = self._getitem_single(index)
        return batch



    def __len__(self):
        return len(self.geodata)
