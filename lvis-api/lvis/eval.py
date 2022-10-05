import datetime
import logging
from collections import OrderedDict, defaultdict
from operator import index

import numpy as np
from lvis.lvis import LVIS
from lvis.results import LVISResults

import pycocotools.mask as mask_utils


lvis_class_distribution = [[12, 13, 16, 19, 20, 29, 30, 37, 38, 39, 41, 48, 50, 51, 62, 68, 70, 77, 81, 84, 92, 104, 105, 112, 116, 118, 122, 125, 129, 130, 135, 139, 141, 143, 146, 150, 154, 158, 160, 163, 166, 171, 178, 181, 195, 201, 208, 209, 213, 214, 221, 222, 230, 232, 233, 235, 236, 237, 239, 243, 244, 246, 249, 250, 256, 257, 261, 264, 265, 268, 269, 274, 280, 281, 286, 290, 291, 293, 294, 299, 300, 301, 303, 306, 309, 312, 315, 316, 320, 322, 325, 330, 332, 347, 348, 351, 352, 353, 354, 356, 361, 363, 364, 365, 367, 373, 375, 380, 381, 387, 388, 396, 397, 399, 404, 406, 409, 412, 413, 415, 419, 425, 426, 427, 430, 431, 434, 438, 445, 448, 455, 457, 466, 477, 478, 479, 480, 481, 485, 487, 490, 491, 502, 505, 507, 508, 512, 515, 517, 526, 531, 534, 537, 540, 541, 542, 544, 550, 556, 559, 560, 566, 567, 570, 571, 573, 574, 576, 579, 581, 582, 584, 593, 596, 598, 601, 602, 605, 609, 615, 617, 618, 619, 624, 631, 633, 634, 637, 639, 645, 647, 650, 656, 661, 662, 663, 664, 670, 671, 673, 677, 685, 687, 689, 690, 692, 701, 709, 711, 713, 721, 726, 728, 729, 732, 742, 751, 753, 754, 757, 758, 763, 768, 771, 777, 778, 782, 783, 784, 786, 787, 791, 795, 802, 804, 807, 808, 809, 811, 814, 819, 821, 822, 823, 828, 830, 848, 849, 850, 851, 852, 854, 855, 857, 858, 861, 863, 868, 872, 882, 885, 886, 889, 890, 891, 893, 901, 904, 907, 912, 913, 916, 917, 919, 924, 930, 936, 937, 938, 940, 941, 943, 944, 951, 955, 957, 968, 971, 973, 974, 982, 984, 986, 989, 990, 991, 993, 997, 1002, 1004, 1009, 1011, 1014, 1015, 1027, 1028, 1029, 1030, 1031, 1046, 1047, 1048, 1052, 1053, 1056, 1057, 1074, 1079, 1083, 1115, 1117, 1118, 1123, 1125, 1128, 1134, 1143, 1144, 1145, 1147, 1149, 1156, 1157, 1158, 1164, 1166, 1192], 
[0, 4, 5, 6, 7, 8, 9, 15, 17, 21, 23, 24, 25, 27, 32, 36, 43, 45, 46, 53, 54, 61, 63, 66, 69, 71, 72, 73, 83, 90, 91, 96, 97, 99, 100, 101, 102, 106, 107, 110, 119, 120, 121, 123, 127, 128, 133, 134, 140, 144, 147, 148, 152, 155, 156, 157, 161, 162, 164, 165, 167, 169, 173, 175, 179, 183, 185, 186, 187, 189, 190, 192, 196, 197, 198, 199, 200, 204, 205, 210, 211, 212, 215, 218, 219, 220, 223, 226, 227, 238, 240, 241, 242, 245, 247, 248, 255, 259, 262, 263, 266, 267, 272, 273, 277, 278, 279, 282, 285, 287, 288, 289, 292, 307, 310, 311, 313, 314, 317, 319, 321, 324, 326, 327, 328, 331, 333, 334, 335, 336, 338, 339, 340, 342, 344, 355, 358, 359, 362, 369, 370, 382, 383, 385, 390, 392, 395, 398, 401, 402, 405, 407, 411, 416, 417, 418, 422, 423, 424, 432, 433, 435, 437, 442, 447, 449, 452, 453, 454, 456, 459, 461, 462, 463, 464, 465, 467, 469, 470, 471, 472, 474, 475, 482, 483, 484, 486, 488, 489, 492, 493, 494, 496, 498, 500, 503, 504, 506, 510, 511, 518, 519, 521, 522, 524, 525, 528, 529, 530, 532, 536, 538, 545, 549, 551, 552, 553, 554, 557, 561, 563, 572, 575, 578, 580, 583, 586, 587, 589, 592, 595, 597, 599, 600, 603, 606, 607, 611, 612, 621, 622, 628, 635, 636, 648, 649, 651, 653, 655, 659, 665, 666, 672, 676, 679, 680, 681, 682, 683, 694, 695, 696, 698, 706, 710, 716, 717, 719, 720, 722, 724, 730, 731, 735, 736, 739, 740, 741, 743, 745, 746, 749, 752, 759, 760, 761, 762, 764, 766, 767, 769, 772, 773, 774, 776, 779, 785, 789, 790, 793, 794, 796, 800, 801, 806, 812, 818, 820, 824, 825, 829, 832, 833, 838, 839, 840, 841, 842, 843, 845, 846, 853, 856
, 860, 862, 865, 866, 867, 869, 870, 871, 873, 874, 876, 877, 878, 881, 883, 887, 888, 892, 894, 896, 900, 905, 906, 908, 921, 925, 927, 928, 929, 931, 932, 933, 934, 935, 939, 945, 949, 953, 959, 962, 969, 970, 972, 976, 977, 983, 987, 988, 995, 996, 998, 1000, 1001, 1003, 1005, 1006, 1008, 1012, 1013, 1021, 1033, 1035, 1037, 1038, 1039, 1040, 1043, 1045, 1050, 1061, 1062, 1064, 1065, 1066, 1067, 1068, 1072, 1075, 1080, 1081, 1084, 1085, 1086, 1087, 1088, 1089, 1091, 1093, 1100, 1105, 1106, 1110, 1112, 1119, 1124, 1126, 1127, 1129, 1130, 1131, 1136, 1137, 1139, 1142, 1146, 1148, 1150, 1151, 1152, 1153, 1159, 1162, 1163, 1165, 1167, 1168, 1169, 1170, 1173, 1174, 1175, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1186, 1188, 1191, 1193, 1194, 1195, 1198, 1199, 1200, 1202], 
[1, 2, 3, 10, 11, 14, 18, 22, 26, 28, 31, 33, 34, 35, 40, 42, 44, 47, 49, 52, 55, 56, 57, 58, 59, 60, 64, 65, 67, 74, 75, 76, 78, 79, 80, 82, 85, 86, 87, 88, 89, 93, 94, 95, 98, 103, 108, 109, 111, 113, 114, 115, 117, 124, 126, 131, 132, 136, 137, 138, 142, 145, 149, 151, 153, 159, 168, 170, 172, 174, 176, 177, 180, 182, 184, 188, 191, 193, 194, 202, 203, 206, 207, 216, 217, 224, 225, 228, 229, 231, 234, 251, 252, 253, 254, 258, 260, 270, 271, 275, 276, 283, 284, 295, 296, 297, 298, 302, 304, 305, 308, 318, 323, 329, 337, 341, 343, 345, 346, 349, 350, 357, 360, 366, 368, 371, 372, 374, 376, 377, 378, 379, 384, 386, 389, 391, 393, 394, 400, 403, 408, 410, 414, 420, 421, 428, 429, 436, 439, 440, 441, 443, 444, 446, 450, 451, 458, 460, 468, 473, 476, 495, 497, 499, 501, 509, 513, 514, 516, 520, 523, 527, 533, 535, 539, 543, 546, 547, 548, 555, 558, 562, 564, 565, 568, 569, 577, 585, 588, 590, 591, 594, 604, 608, 610, 613, 614, 616, 620, 623, 625, 626, 627, 629, 630, 632, 638, 640, 641, 642, 643, 644, 646, 652, 654, 657, 658, 660, 667, 668, 669, 674, 675, 678, 684, 686, 688, 691, 693, 697, 699, 700, 702, 703, 704, 705, 707, 708, 712, 714, 715, 718, 723, 725, 727, 733, 734, 737, 738, 744, 747, 748, 750, 755, 756, 765, 770, 775, 780, 781, 788, 792, 797, 798, 799, 803, 805, 810, 813, 815, 816, 817, 826, 827, 831, 834, 835, 836, 837, 844, 847, 859, 864, 875, 879, 880, 884, 895, 897, 898, 899, 902, 903, 909, 910, 911, 914, 915, 918, 920, 922, 923, 926, 942, 946, 947, 948, 950, 952, 954, 956, 958, 960, 961, 963, 964, 965, 966, 967, 975, 978, 979, 980, 981, 985, 992, 994, 999, 1007, 1010, 1016, 1017, 1018, 1019, 1020, 1022, 1023, 1024, 1025, 1026, 1032, 1034, 1036, 1041, 1042, 1044, 1049, 1051, 1054, 1055, 1058, 1059, 1060, 1063, 1069, 1070, 1071, 1073, 1076, 1077, 1078, 1082, 1090, 1092, 1094, 1095, 1096, 1097, 1098, 1099, 1101, 1102, 1103, 1104, 1107, 1108, 1109, 1111, 1113, 1114, 1116, 1120, 1121, 1122, 1132, 1133, 1135, 1138, 1140, 1141, 1154, 1155, 1160, 1161, 1171, 1172, 1176, 1177, 1185, 1187, 1189, 1190, 1196, 1197, 1201]]

class_distribution =np.array([np.array(xi) for xi in lvis_class_distribution])

def index_to_freq_group(idx):
    if (np.where(class_distribution[0] == idx)[0]).shape[0] > 0:
        return "rare"
    elif (np.where(class_distribution[1] == idx)[0]).shape[0] > 0:
        return "common"
    elif (np.where(class_distribution[2] == idx)[0]).shape[0] > 0:
        return "frequent"
    else:
        return "unknown"

CLASSES = (
    'aerosol_can', 'air_conditioner', 'airplane', 'alarm_clock', 'alcohol',
    'alligator', 'almond', 'ambulance', 'amplifier', 'anklet', 'antenna',
    'apple', 'applesauce', 'apricot', 'apron', 'aquarium',
    'arctic_(type_of_shoe)', 'armband', 'armchair', 'armoire', 'armor',
    'artichoke', 'trash_can', 'ashtray', 'asparagus', 'atomizer',
    'avocado', 'award', 'awning', 'ax', 'baboon', 'baby_buggy',
    'basketball_backboard', 'backpack', 'handbag', 'suitcase', 'bagel',
    'bagpipe', 'baguet', 'bait', 'ball', 'ballet_skirt', 'balloon',
    'bamboo', 'banana', 'Band_Aid', 'bandage', 'bandanna', 'banjo',
    'banner', 'barbell', 'barge', 'barrel', 'barrette', 'barrow',
    'baseball_base', 'baseball', 'baseball_bat', 'baseball_cap',
    'baseball_glove', 'basket', 'basketball', 'bass_horn', 'bat_(animal)',
    'bath_mat', 'bath_towel', 'bathrobe', 'bathtub', 'batter_(food)',
    'battery', 'beachball', 'bead', 'bean_curd', 'beanbag', 'beanie',
    'bear', 'bed', 'bedpan', 'bedspread', 'cow', 'beef_(food)', 'beeper',
    'beer_bottle', 'beer_can', 'beetle', 'bell', 'bell_pepper', 'belt',
    'belt_buckle', 'bench', 'beret', 'bib', 'Bible', 'bicycle', 'visor',
    'billboard', 'binder', 'binoculars', 'bird', 'birdfeeder', 'birdbath',
    'birdcage', 'birdhouse', 'birthday_cake', 'birthday_card',
    'pirate_flag', 'black_sheep', 'blackberry', 'blackboard', 'blanket',
    'blazer', 'blender', 'blimp', 'blinker', 'blouse', 'blueberry',
    'gameboard', 'boat', 'bob', 'bobbin', 'bobby_pin', 'boiled_egg',
    'bolo_tie', 'deadbolt', 'bolt', 'bonnet', 'book', 'bookcase',
    'booklet', 'bookmark', 'boom_microphone', 'boot', 'bottle',
    'bottle_opener', 'bouquet', 'bow_(weapon)', 'bow_(decorative_ribbons)',
    'bow-tie', 'bowl', 'pipe_bowl', 'bowler_hat', 'bowling_ball', 'box',
    'boxing_glove', 'suspenders', 'bracelet', 'brass_plaque', 'brassiere',
    'bread-bin', 'bread', 'breechcloth', 'bridal_gown', 'briefcase',
    'broccoli', 'broach', 'broom', 'brownie', 'brussels_sprouts',
    'bubble_gum', 'bucket', 'horse_buggy', 'bull', 'bulldog', 'bulldozer',
    'bullet_train', 'bulletin_board', 'bulletproof_vest', 'bullhorn',
    'bun', 'bunk_bed', 'buoy', 'burrito', 'bus_(vehicle)', 'business_card',
    'butter', 'butterfly', 'button', 'cab_(taxi)', 'cabana', 'cabin_car',
    'cabinet', 'locker', 'cake', 'calculator', 'calendar', 'calf',
    'camcorder', 'camel', 'camera', 'camera_lens', 'camper_(vehicle)',
    'can', 'can_opener', 'candle', 'candle_holder', 'candy_bar',
    'candy_cane', 'walking_cane', 'canister', 'canoe', 'cantaloup',
    'canteen', 'cap_(headwear)', 'bottle_cap', 'cape', 'cappuccino',
    'car_(automobile)', 'railcar_(part_of_a_train)', 'elevator_car',
    'car_battery', 'identity_card', 'card', 'cardigan', 'cargo_ship',
    'carnation', 'horse_carriage', 'carrot', 'tote_bag', 'cart', 'carton',
    'cash_register', 'casserole', 'cassette', 'cast', 'cat', 'cauliflower',
    'cayenne_(spice)', 'CD_player', 'celery', 'cellular_telephone',
    'chain_mail', 'chair', 'chaise_longue', 'chalice', 'chandelier',
    'chap', 'checkbook', 'checkerboard', 'cherry', 'chessboard',
    'chicken_(animal)', 'chickpea', 'chili_(vegetable)', 'chime',
    'chinaware', 'crisp_(potato_chip)', 'poker_chip', 'chocolate_bar',
    'chocolate_cake', 'chocolate_milk', 'chocolate_mousse', 'choker',
    'chopping_board', 'chopstick', 'Christmas_tree', 'slide', 'cider',
    'cigar_box', 'cigarette', 'cigarette_case', 'cistern', 'clarinet',
    'clasp', 'cleansing_agent', 'cleat_(for_securing_rope)', 'clementine',
    'clip', 'clipboard', 'clippers_(for_plants)', 'cloak', 'clock',
    'clock_tower', 'clothes_hamper', 'clothespin', 'clutch_bag', 'coaster',
    'coat', 'coat_hanger', 'coatrack', 'cock', 'cockroach',
    'cocoa_(beverage)', 'coconut', 'coffee_maker', 'coffee_table',
    'coffeepot', 'coil', 'coin', 'colander', 'coleslaw',
    'coloring_material', 'combination_lock', 'pacifier', 'comic_book',
    'compass', 'computer_keyboard', 'condiment', 'cone', 'control',
    'convertible_(automobile)', 'sofa_bed', 'cooker', 'cookie',
    'cooking_utensil', 'cooler_(for_food)', 'cork_(bottle_plug)',
    'corkboard', 'corkscrew', 'edible_corn', 'cornbread', 'cornet',
    'cornice', 'cornmeal', 'corset', 'costume', 'cougar', 'coverall',
    'cowbell', 'cowboy_hat', 'crab_(animal)', 'crabmeat', 'cracker',
    'crape', 'crate', 'crayon', 'cream_pitcher', 'crescent_roll', 'crib',
    'crock_pot', 'crossbar', 'crouton', 'crow', 'crowbar', 'crown',
    'crucifix', 'cruise_ship', 'police_cruiser', 'crumb', 'crutch',
    'cub_(animal)', 'cube', 'cucumber', 'cufflink', 'cup', 'trophy_cup',
    'cupboard', 'cupcake', 'hair_curler', 'curling_iron', 'curtain',
    'cushion', 'cylinder', 'cymbal', 'dagger', 'dalmatian', 'dartboard',
    'date_(fruit)', 'deck_chair', 'deer', 'dental_floss', 'desk',
    'detergent', 'diaper', 'diary', 'die', 'dinghy', 'dining_table', 'tux',
    'dish', 'dish_antenna', 'dishrag', 'dishtowel', 'dishwasher',
    'dishwasher_detergent', 'dispenser', 'diving_board', 'Dixie_cup',
    'dog', 'dog_collar', 'doll', 'dollar', 'dollhouse', 'dolphin',
    'domestic_ass', 'doorknob', 'doormat', 'doughnut', 'dove', 'dragonfly',
    'drawer', 'underdrawers', 'dress', 'dress_hat', 'dress_suit',
    'dresser', 'drill', 'drone', 'dropper', 'drum_(musical_instrument)',
    'drumstick', 'duck', 'duckling', 'duct_tape', 'duffel_bag', 'dumbbell',
    'dumpster', 'dustpan', 'eagle', 'earphone', 'earplug', 'earring',
    'easel', 'eclair', 'eel', 'egg', 'egg_roll', 'egg_yolk', 'eggbeater',
    'eggplant', 'electric_chair', 'refrigerator', 'elephant', 'elk',
    'envelope', 'eraser', 'escargot', 'eyepatch', 'falcon', 'fan',
    'faucet', 'fedora', 'ferret', 'Ferris_wheel', 'ferry', 'fig_(fruit)',
    'fighter_jet', 'figurine', 'file_cabinet', 'file_(tool)', 'fire_alarm',
    'fire_engine', 'fire_extinguisher', 'fire_hose', 'fireplace',
    'fireplug', 'first-aid_kit', 'fish', 'fish_(food)', 'fishbowl',
    'fishing_rod', 'flag', 'flagpole', 'flamingo', 'flannel', 'flap',
    'flash', 'flashlight', 'fleece', 'flip-flop_(sandal)',
    'flipper_(footwear)', 'flower_arrangement', 'flute_glass', 'foal',
    'folding_chair', 'food_processor', 'football_(American)',
    'football_helmet', 'footstool', 'fork', 'forklift', 'freight_car',
    'French_toast', 'freshener', 'frisbee', 'frog', 'fruit_juice',
    'frying_pan', 'fudge', 'funnel', 'futon', 'gag', 'garbage',
    'garbage_truck', 'garden_hose', 'gargle', 'gargoyle', 'garlic',
    'gasmask', 'gazelle', 'gelatin', 'gemstone', 'generator',
    'giant_panda', 'gift_wrap', 'ginger', 'giraffe', 'cincture',
    'glass_(drink_container)', 'globe', 'glove', 'goat', 'goggles',
    'goldfish', 'golf_club', 'golfcart', 'gondola_(boat)', 'goose',
    'gorilla', 'gourd', 'grape', 'grater', 'gravestone', 'gravy_boat',
    'green_bean', 'green_onion', 'griddle', 'grill', 'grits', 'grizzly',
    'grocery_bag', 'guitar', 'gull', 'gun', 'hairbrush', 'hairnet',
    'hairpin', 'halter_top', 'ham', 'hamburger', 'hammer', 'hammock',
    'hamper', 'hamster', 'hair_dryer', 'hand_glass', 'hand_towel',
    'handcart', 'handcuff', 'handkerchief', 'handle', 'handsaw',
    'hardback_book', 'harmonium', 'hat', 'hatbox', 'veil', 'headband',
    'headboard', 'headlight', 'headscarf', 'headset',
    'headstall_(for_horses)', 'heart', 'heater', 'helicopter', 'helmet',
    'heron', 'highchair', 'hinge', 'hippopotamus', 'hockey_stick', 'hog',
    'home_plate_(baseball)', 'honey', 'fume_hood', 'hook', 'hookah',
    'hornet', 'horse', 'hose', 'hot-air_balloon', 'hotplate', 'hot_sauce',
    'hourglass', 'houseboat', 'hummingbird', 'hummus', 'polar_bear',
    'icecream', 'popsicle', 'ice_maker', 'ice_pack', 'ice_skate',
    'igniter', 'inhaler', 'iPod', 'iron_(for_clothing)', 'ironing_board',
    'jacket', 'jam', 'jar', 'jean', 'jeep', 'jelly_bean', 'jersey',
    'jet_plane', 'jewel', 'jewelry', 'joystick', 'jumpsuit', 'kayak',
    'keg', 'kennel', 'kettle', 'key', 'keycard', 'kilt', 'kimono',
    'kitchen_sink', 'kitchen_table', 'kite', 'kitten', 'kiwi_fruit',
    'knee_pad', 'knife', 'knitting_needle', 'knob', 'knocker_(on_a_door)',
    'koala', 'lab_coat', 'ladder', 'ladle', 'ladybug', 'lamb_(animal)',
    'lamb-chop', 'lamp', 'lamppost', 'lampshade', 'lantern', 'lanyard',
    'laptop_computer', 'lasagna', 'latch', 'lawn_mower', 'leather',
    'legging_(clothing)', 'Lego', 'legume', 'lemon', 'lemonade', 'lettuce',
    'license_plate', 'life_buoy', 'life_jacket', 'lightbulb',
    'lightning_rod', 'lime', 'limousine', 'lion', 'lip_balm', 'liquor',
    'lizard', 'log', 'lollipop', 'speaker_(stereo_equipment)', 'loveseat',
    'machine_gun', 'magazine', 'magnet', 'mail_slot', 'mailbox_(at_home)',
    'mallard', 'mallet', 'mammoth', 'manatee', 'mandarin_orange', 'manger',
    'manhole', 'map', 'marker', 'martini', 'mascot', 'mashed_potato',
    'masher', 'mask', 'mast', 'mat_(gym_equipment)', 'matchbox',
    'mattress', 'measuring_cup', 'measuring_stick', 'meatball', 'medicine',
    'melon', 'microphone', 'microscope', 'microwave_oven', 'milestone',
    'milk', 'milk_can', 'milkshake', 'minivan', 'mint_candy', 'mirror',
    'mitten', 'mixer_(kitchen_tool)', 'money',
    'monitor_(computer_equipment) computer_monitor', 'monkey', 'motor',
    'motor_scooter', 'motor_vehicle', 'motorcycle', 'mound_(baseball)',
    'mouse_(computer_equipment)', 'mousepad', 'muffin', 'mug', 'mushroom',
    'music_stool', 'musical_instrument', 'nailfile', 'napkin',
    'neckerchief', 'necklace', 'necktie', 'needle', 'nest', 'newspaper',
    'newsstand', 'nightshirt', 'nosebag_(for_animals)',
    'noseband_(for_animals)', 'notebook', 'notepad', 'nut', 'nutcracker',
    'oar', 'octopus_(food)', 'octopus_(animal)', 'oil_lamp', 'olive_oil',
    'omelet', 'onion', 'orange_(fruit)', 'orange_juice', 'ostrich',
    'ottoman', 'oven', 'overalls_(clothing)', 'owl', 'packet', 'inkpad',
    'pad', 'paddle', 'padlock', 'paintbrush', 'painting', 'pajamas',
    'palette', 'pan_(for_cooking)', 'pan_(metal_container)', 'pancake',
    'pantyhose', 'papaya', 'paper_plate', 'paper_towel', 'paperback_book',
    'paperweight', 'parachute', 'parakeet', 'parasail_(sports)', 'parasol',
    'parchment', 'parka', 'parking_meter', 'parrot',
    'passenger_car_(part_of_a_train)', 'passenger_ship', 'passport',
    'pastry', 'patty_(food)', 'pea_(food)', 'peach', 'peanut_butter',
    'pear', 'peeler_(tool_for_fruit_and_vegetables)', 'wooden_leg',
    'pegboard', 'pelican', 'pen', 'pencil', 'pencil_box',
    'pencil_sharpener', 'pendulum', 'penguin', 'pennant', 'penny_(coin)',
    'pepper', 'pepper_mill', 'perfume', 'persimmon', 'person', 'pet',
    'pew_(church_bench)', 'phonebook', 'phonograph_record', 'piano',
    'pickle', 'pickup_truck', 'pie', 'pigeon', 'piggy_bank', 'pillow',
    'pin_(non_jewelry)', 'pineapple', 'pinecone', 'ping-pong_ball',
    'pinwheel', 'tobacco_pipe', 'pipe', 'pistol', 'pita_(bread)',
    'pitcher_(vessel_for_liquid)', 'pitchfork', 'pizza', 'place_mat',
    'plate', 'platter', 'playpen', 'pliers', 'plow_(farm_equipment)',
    'plume', 'pocket_watch', 'pocketknife', 'poker_(fire_stirring_tool)',
    'pole', 'polo_shirt', 'poncho', 'pony', 'pool_table', 'pop_(soda)',
    'postbox_(public)', 'postcard', 'poster', 'pot', 'flowerpot', 'potato',
    'potholder', 'pottery', 'pouch', 'power_shovel', 'prawn', 'pretzel',
    'printer', 'projectile_(weapon)', 'projector', 'propeller', 'prune',
    'pudding', 'puffer_(fish)', 'puffin', 'pug-dog', 'pumpkin', 'puncher',
    'puppet', 'puppy', 'quesadilla', 'quiche', 'quilt', 'rabbit',
    'race_car', 'racket', 'radar', 'radiator', 'radio_receiver', 'radish',
    'raft', 'rag_doll', 'raincoat', 'ram_(animal)', 'raspberry', 'rat',
    'razorblade', 'reamer_(juicer)', 'rearview_mirror', 'receipt',
    'recliner', 'record_player', 'reflector', 'remote_control',
    'rhinoceros', 'rib_(food)', 'rifle', 'ring', 'river_boat', 'road_map',
    'robe', 'rocking_chair', 'rodent', 'roller_skate', 'Rollerblade',
    'rolling_pin', 'root_beer', 'router_(computer_equipment)',
    'rubber_band', 'runner_(carpet)', 'plastic_bag',
    'saddle_(on_an_animal)', 'saddle_blanket', 'saddlebag', 'safety_pin',
    'sail', 'salad', 'salad_plate', 'salami', 'salmon_(fish)',
    'salmon_(food)', 'salsa', 'saltshaker', 'sandal_(type_of_shoe)',
    'sandwich', 'satchel', 'saucepan', 'saucer', 'sausage', 'sawhorse',
    'saxophone', 'scale_(measuring_instrument)', 'scarecrow', 'scarf',
    'school_bus', 'scissors', 'scoreboard', 'scraper', 'screwdriver',
    'scrubbing_brush', 'sculpture', 'seabird', 'seahorse', 'seaplane',
    'seashell', 'sewing_machine', 'shaker', 'shampoo', 'shark',
    'sharpener', 'Sharpie', 'shaver_(electric)', 'shaving_cream', 'shawl',
    'shears', 'sheep', 'shepherd_dog', 'sherbert', 'shield', 'shirt',
    'shoe', 'shopping_bag', 'shopping_cart', 'short_pants', 'shot_glass',
    'shoulder_bag', 'shovel', 'shower_head', 'shower_cap',
    'shower_curtain', 'shredder_(for_paper)', 'signboard', 'silo', 'sink',
    'skateboard', 'skewer', 'ski', 'ski_boot', 'ski_parka', 'ski_pole',
    'skirt', 'skullcap', 'sled', 'sleeping_bag', 'sling_(bandage)',
    'slipper_(footwear)', 'smoothie', 'snake', 'snowboard', 'snowman',
    'snowmobile', 'soap', 'soccer_ball', 'sock', 'sofa', 'softball',
    'solar_array', 'sombrero', 'soup', 'soup_bowl', 'soupspoon',
    'sour_cream', 'soya_milk', 'space_shuttle', 'sparkler_(fireworks)',
    'spatula', 'spear', 'spectacles', 'spice_rack', 'spider', 'crawfish',
    'sponge', 'spoon', 'sportswear', 'spotlight', 'squid_(food)',
    'squirrel', 'stagecoach', 'stapler_(stapling_machine)', 'starfish',
    'statue_(sculpture)', 'steak_(food)', 'steak_knife', 'steering_wheel',
    'stepladder', 'step_stool', 'stereo_(sound_system)', 'stew', 'stirrer',
    'stirrup', 'stool', 'stop_sign', 'brake_light', 'stove', 'strainer',
    'strap', 'straw_(for_drinking)', 'strawberry', 'street_sign',
    'streetlight', 'string_cheese', 'stylus', 'subwoofer', 'sugar_bowl',
    'sugarcane_(plant)', 'suit_(clothing)', 'sunflower', 'sunglasses',
    'sunhat', 'surfboard', 'sushi', 'mop', 'sweat_pants', 'sweatband',
    'sweater', 'sweatshirt', 'sweet_potato', 'swimsuit', 'sword',
    'syringe', 'Tabasco_sauce', 'table-tennis_table', 'table',
    'table_lamp', 'tablecloth', 'tachometer', 'taco', 'tag', 'taillight',
    'tambourine', 'army_tank', 'tank_(storage_vessel)',
    'tank_top_(clothing)', 'tape_(sticky_cloth_or_paper)', 'tape_measure',
    'tapestry', 'tarp', 'tartan', 'tassel', 'tea_bag', 'teacup',
    'teakettle', 'teapot', 'teddy_bear', 'telephone', 'telephone_booth',
    'telephone_pole', 'telephoto_lens', 'television_camera',
    'television_set', 'tennis_ball', 'tennis_racket', 'tequila',
    'thermometer', 'thermos_bottle', 'thermostat', 'thimble', 'thread',
    'thumbtack', 'tiara', 'tiger', 'tights_(clothing)', 'timer', 'tinfoil',
    'tinsel', 'tissue_paper', 'toast_(food)', 'toaster', 'toaster_oven',
    'toilet', 'toilet_tissue', 'tomato', 'tongs', 'toolbox', 'toothbrush',
    'toothpaste', 'toothpick', 'cover', 'tortilla', 'tow_truck', 'towel',
    'towel_rack', 'toy', 'tractor_(farm_equipment)', 'traffic_light',
    'dirt_bike', 'trailer_truck', 'train_(railroad_vehicle)', 'trampoline',
    'tray', 'trench_coat', 'triangle_(musical_instrument)', 'tricycle',
    'tripod', 'trousers', 'truck', 'truffle_(chocolate)', 'trunk', 'vat',
    'turban', 'turkey_(food)', 'turnip', 'turtle', 'turtleneck_(clothing)',
    'typewriter', 'umbrella', 'underwear', 'unicycle', 'urinal', 'urn',
    'vacuum_cleaner', 'vase', 'vending_machine', 'vent', 'vest',
    'videotape', 'vinegar', 'violin', 'vodka', 'volleyball', 'vulture',
    'waffle', 'waffle_iron', 'wagon', 'wagon_wheel', 'walking_stick',
    'wall_clock', 'wall_socket', 'wallet', 'walrus', 'wardrobe',
    'washbasin', 'automatic_washer', 'watch', 'water_bottle',
    'water_cooler', 'water_faucet', 'water_heater', 'water_jug',
    'water_gun', 'water_scooter', 'water_ski', 'water_tower',
    'watering_can', 'watermelon', 'weathervane', 'webcam', 'wedding_cake',
    'wedding_ring', 'wet_suit', 'wheel', 'wheelchair', 'whipped_cream',
    'whistle', 'wig', 'wind_chime', 'windmill', 'window_box_(for_plants)',
    'windshield_wiper', 'windsock', 'wine_bottle', 'wine_bucket',
    'wineglass', 'blinder_(for_horses)', 'wok', 'wolf', 'wooden_spoon',
    'wreath', 'wrench', 'wristband', 'wristlet', 'yacht', 'yogurt',
    'yoke_(animal_equipment)', 'zebra', 'zucchini')


class LVISEval:
    def __init__(self, lvis_gt, lvis_dt, iou_type="segm"):
        """Constructor for LVISEval.
        Args:
            lvis_gt (LVIS class instance,
                or str containing path of annotation file)
            lvis_dt (LVISResult class instance,
                or str containing path of result file,
            or list of dict)
            iou_type (str): segm or bbox evaluation
        """
        self.logger = logging.getLogger(__name__)

        if iou_type not in ["bbox", "segm"]:
            raise ValueError("iou_type: {} is not supported.".format(iou_type))

        if isinstance(lvis_gt, LVIS):
            self.lvis_gt = lvis_gt
        elif isinstance(lvis_gt, str):
            self.lvis_gt = LVIS(lvis_gt)
        else:
            raise TypeError("Unsupported type {} of lvis_gt.".format(lvis_gt))

        if isinstance(lvis_dt, LVISResults):
            self.lvis_dt = lvis_dt
        elif isinstance(lvis_dt, (str, list)):
            self.lvis_dt = LVISResults(self.lvis_gt, lvis_dt)
        else:
            raise TypeError("Unsupported type {} of lvis_dt.".format(lvis_dt))

        # per-image per-category evaluation results
        self.eval_imgs = defaultdict(list)
        self.eval = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self.params = Params(iou_type=iou_type)  # parameters
        self.results = OrderedDict()
        self.ious = {}  # ious between all gts and dts

        self.params.img_ids = sorted(self.lvis_gt.get_img_ids())
        self.params.cat_ids = sorted(self.lvis_gt.get_cat_ids())

    def _to_mask(self, anns, lvis):
        for ann in anns:
            rle = lvis.ann_to_rle(ann)
            ann["segmentation"] = rle

    def _prepare(self):
        """Prepare self._gts and self._dts for evaluation based on params."""

        cat_ids = self.params.cat_ids if self.params.cat_ids else None

        gts = self.lvis_gt.load_anns(
            self.lvis_gt.get_ann_ids(img_ids=self.params.img_ids,
                                     cat_ids=cat_ids))
        dts = self.lvis_dt.load_anns(
            self.lvis_dt.get_ann_ids(img_ids=self.params.img_ids,
                                     cat_ids=cat_ids))
        # convert ground truth to mask if iou_type == 'segm'
        if self.params.iou_type == "segm":
            self._to_mask(gts, self.lvis_gt)
            self._to_mask(dts, self.lvis_dt)

        # set ignore flag
        for gt in gts:
            if "ignore" not in gt:
                gt["ignore"] = 0

        for gt in gts:
            self._gts[gt["image_id"], gt["category_id"]].append(gt)

        # For federated dataset evaluation we will filter out all dt for an
        # image which belong to categories not present in gt and not present in
        # the negative list for an image. In other words detector is not
        # penalized for categories about which we don't have gt information
        # about their presence or absence in an image.
        img_data = self.lvis_gt.load_imgs(ids=self.params.img_ids)
        # per image map of categories not present in image
        img_nl = {d["id"]: d["neg_category_ids"] for d in img_data}
        # per image list of categories present in image
        img_pl = defaultdict(set)
        for ann in gts:
            img_pl[ann["image_id"]].add(ann["category_id"])
        # per image map of categoires which have missing gt. For these
        # categories we don't penalize the detector for flase positives.
        self.img_nel = {
            d["id"]: d["not_exhaustive_category_ids"]
            for d in img_data
        }
        # print(f"img_nl: {img_nl}")
        # print(f"img_pl: {img_pl}")
        # cat_name_num_det_0 = defaultdict(int)
        # not_present_counts = defaultdict(int)
        # for dt in dts:
        #     img_id, cat_id = dt["image_id"], dt["category_id"]
        #     cat_name_num_det_0[index_to_freq_group(cat_id-1)] += 1
        #     if cat_id not in img_nl[img_id] and cat_id not in img_pl[img_id]:
        #         not_present_counts[index_to_freq_group(cat_id-1)] += 1
        #         continue
        #     self._dts[img_id, cat_id].append(dt)
        # print(f"not_present_counts: {not_present_counts}")
        # print(f"cat_name_num_det_0: {cat_name_num_det_0}")


        for dt in dts:
            img_id, cat_id = dt["image_id"], dt["category_id"]
            if cat_id not in img_nl[img_id] and cat_id not in img_pl[img_id]:
                continue
            self._dts[img_id, cat_id].append(dt)

        self.freq_groups = self._prepare_freq_group()

    def _prepare_freq_group(self):
        freq_groups = [[] for _ in self.params.img_count_lbl]
        cat_data = self.lvis_gt.load_cats(self.params.cat_ids)
        for idx, _cat_data in enumerate(cat_data):
            frequency = _cat_data["frequency"]
            freq_groups[self.params.img_count_lbl.index(frequency)].append(idx)
        return freq_groups

    def debug(self):
      self._prepare()
      img_id_det_len_map = defaultdict(int)
      for img_id in self.params.img_ids:
        for cat_id in self.params.cat_ids:
          gt, dt = self._get_gt_dt(img_id, cat_id)
          img_id_det_len_map[img_id] += len(dt)
        if img_id_det_len_map[img_id] > 300:
          print(f"img with id {img_id} has more than 300 dets")
      print(img_id_det_len_map)

    def evaluate(self):
        """
        Run per image evaluation on given images and store results
        (a list of dict) in self.eval_imgs.
        """
        self.logger.info("Running per image evaluation.")
        self.logger.info("Evaluate annotation type *{}*".format(
            self.params.iou_type))

        self.params.img_ids = list(np.unique(self.params.img_ids))

        if self.params.use_cats:
            cat_ids = self.params.cat_ids
        else:
            cat_ids = [-1]

        self._prepare()

        self.ious = {(img_id, cat_id): self.compute_iou(img_id, cat_id)
                     for img_id in self.params.img_ids for cat_id in cat_ids}

        # loop through images, area range, max detection number
        self.eval_imgs = [
            self.evaluate_img(img_id, cat_id, area_rng) for cat_id in cat_ids
            for area_rng in self.params.area_rng
            for img_id in self.params.img_ids
        ]

    def _get_gt_dt(self, img_id, cat_id):
        """Create gt, dt which are list of anns/dets. If use_cats is true
        only anns/dets corresponding to tuple (img_id, cat_id) will be
        used. Else, all anns/dets in image are used and cat_id is not used.
        """
        if self.params.use_cats:
            gt = self._gts[img_id, cat_id]
            dt = self._dts[img_id, cat_id]
        else:
            gt = [
                _ann for _cat_id in self.params.cat_ids
                for _ann in self._gts[img_id, cat_id]
            ]
            dt = [
                _ann for _cat_id in self.params.cat_ids
                for _ann in self._dts[img_id, cat_id]
            ]
        return gt, dt

    def compute_iou(self, img_id, cat_id):
        gt, dt = self._get_gt_dt(img_id, cat_id)

        if len(gt) == 0 and len(dt) == 0:
            return []

        # Sort detections in decreasing order of score.
        idx = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in idx]

        iscrowd = [int(False)] * len(gt)

        if self.params.iou_type == "segm":
            ann_type = "segmentation"
        elif self.params.iou_type == "bbox":
            ann_type = "bbox"
        else:
            raise ValueError("Unknown iou_type for iou computation.")
        gt = [g[ann_type] for g in gt]
        dt = [d[ann_type] for d in dt]

        # compute iou between each dt and gt region
        # will return array of shape len(dt), len(gt)
        ious = mask_utils.iou(dt, gt, iscrowd)
        return ious

    def evaluate_img(self, img_id, cat_id, area_rng):
        """Perform evaluation for single category and image."""
        gt, dt = self._get_gt_dt(img_id, cat_id)

        if len(gt) == 0 and len(dt) == 0:
            return None

        # Add another filed _ignore to only consider anns based on area range.
        for g in gt:
            if g["ignore"] or (g["area"] < area_rng[0]
                               or g["area"] > area_rng[1]):
                g["_ignore"] = 1
            else:
                g["_ignore"] = 0

        # Sort gt ignore last
        gt_idx = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
        gt = [gt[i] for i in gt_idx]

        # Sort dt highest score first
        dt_idx = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in dt_idx]

        # load computed ious
        ious = (self.ious[img_id, cat_id][:, gt_idx]
                if len(self.ious[img_id, cat_id]) > 0 else self.ious[img_id,
                                                                     cat_id])

        num_thrs = len(self.params.iou_thrs)
        num_gt = len(gt)
        num_dt = len(dt)

        # Array to store the "id" of the matched dt/gt
        gt_m = np.zeros((num_thrs, num_gt))
        dt_m = np.zeros((num_thrs, num_dt))

        gt_ig = np.array([g["_ignore"] for g in gt])
        dt_ig = np.zeros((num_thrs, num_dt))
        dt_iou = np.zeros((num_thrs, num_dt))

        for iou_thr_idx, iou_thr in enumerate(self.params.iou_thrs):
            if len(ious) == 0:
                break

            for dt_idx, _dt in enumerate(dt):
                iou = min([iou_thr, 1 - 1e-10])
                # information about best match so far (m=-1 -> unmatched)
                # store the gt_idx which matched for _dt
                m = -1
                for gt_idx, _ in enumerate(gt):
                    # if this gt already matched continue
                    if gt_m[iou_thr_idx, gt_idx] > 0:
                        continue
                    # if _dt matched to reg gt, and on ignore gt, stop
                    if m > -1 and gt_ig[m] == 0 and gt_ig[gt_idx] == 1:
                        break
                    # continue to next gt unless better match made
                    if ious[dt_idx, gt_idx] < iou:
                        continue
                    # if match successful and best so far, store appropriately
                    iou = ious[dt_idx, gt_idx]
                    m = gt_idx

                # No match found for _dt, go to next _dt
                if m == -1:
                    continue

                # if gt to ignore for some reason update dt_ig.
                # Should not be used in evaluation.
                dt_ig[iou_thr_idx, dt_idx] = gt_ig[m]
                # _dt match found, update gt_m, and dt_m with "id"
                dt_m[iou_thr_idx, dt_idx] = gt[m]["id"]
                gt_m[iou_thr_idx, m] = _dt["id"]
                dt_iou[iou_thr_idx, dt_idx] = iou

        # For LVIS we will ignore any unmatched detection if that category was
        # not exhaustively annotated in gt.
        dt_ig_mask = [
            d["area"] < area_rng[0] or d["area"] > area_rng[1]
            or d["category_id"] in self.img_nel[d["image_id"]] for d in dt
        ]
        dt_ig_mask = np.array(dt_ig_mask).reshape((1, num_dt))  # 1 X num_dt
        dt_ig_mask = np.repeat(dt_ig_mask, num_thrs, 0)  # num_thrs X num_dt
        # Based on dt_ig_mask ignore any unmatched detection by updating dt_ig
        dt_ig = np.logical_or(dt_ig, np.logical_and(dt_m == 0, dt_ig_mask))
        # store results for given image and category
        return {
            "image_id": img_id,
            "category_id": cat_id,
            "area_rng": area_rng,
            "dt_ids": [d["id"] for d in dt],
            "gt_ids": [g["id"] for g in gt],
            "dt_matches": dt_m,
            "gt_matches": gt_m,
            "dt_scores": [d["score"] for d in dt],
            "gt_ignore": gt_ig,
            "dt_ignore": dt_ig,
            "dt_ious": dt_iou,
        }

    def accumulate(self):
        """Accumulate per image evaluation results and store the result in
        self.eval.
        """
        self.logger.info("Accumulating evaluation results.")

        if not self.eval_imgs:
            self.logger.warn("Please run evaluate first.")

        if self.params.use_cats:
            cat_ids = self.params.cat_ids
        else:
            cat_ids = [-1]

        num_thrs = len(self.params.iou_thrs)
        num_recalls = len(self.params.rec_thrs)
        num_cats = len(cat_ids)
        num_area_rngs = len(self.params.area_rng)
        num_imgs = len(self.params.img_ids)

        # -1 for absent categories
        precision = -np.ones((num_thrs, num_recalls, num_cats, num_area_rngs))
        recall = -np.ones((num_thrs, num_cats, num_area_rngs))

        olrp_loc = -np.ones((num_cats, num_area_rngs))
        olrp_fp = -np.ones((num_cats, num_area_rngs))
        olrp_fn = -np.ones((num_cats, num_area_rngs))
        olrp = -np.ones((num_cats, num_area_rngs))
        lrp_opt_thr = -np.ones((num_cats, num_area_rngs))
        _lrps = {}
        _dt_scores = {}
        _dt_m = {}
        _tps = {}
        _fps = {}
        # Initialize dt_pointers
        dt_pointers = {}
        for cat_idx in range(num_cats):
            dt_pointers[cat_idx] = {}
            for area_idx in range(num_area_rngs):
                dt_pointers[cat_idx][area_idx] = {}

        # Per category evaluation
        for cat_idx in range(num_cats):
            Nk = cat_idx * num_area_rngs * num_imgs
            for area_idx in range(num_area_rngs):
                Na = area_idx * num_imgs
                E = [
                    self.eval_imgs[Nk + Na + img_idx]
                    for img_idx in range(num_imgs)
                ]
                # Remove elements which are None
                E = [e for e in E if e is not None]
                if len(E) == 0:
                    continue

                # Append all scores: shape (N,)
                dt_scores = np.concatenate([e["dt_scores"] for e in E], axis=0)
                dt_ids = np.concatenate([e["dt_ids"] for e in E], axis=0)

                dt_idx = np.argsort(-dt_scores, kind="mergesort")
                dt_scores = dt_scores[dt_idx]
                dt_ids = dt_ids[dt_idx]

                dt_m = np.concatenate([e["dt_matches"] for e in E],
                                      axis=1)[:, dt_idx]
                dt_ig = np.concatenate([e["dt_ignore"] for e in E],
                                       axis=1)[:, dt_idx]
                dt_iou = np.concatenate([e["dt_ious"] for e in E],
                                        axis=1)[:, dt_idx]

                gt_ig = np.concatenate([e["gt_ignore"] for e in E])
                # num gt anns to consider
                num_gt = np.count_nonzero(gt_ig == 0)

                if num_gt == 0:
                    continue

                tps = np.logical_and(dt_m, np.logical_not(dt_ig))
                fps = np.logical_and(np.logical_not(dt_m),
                                     np.logical_not(dt_ig))

                dt_iou = np.multiply(dt_iou, tps)
                tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)

                dt_pointers[cat_idx][area_idx] = {
                    "dt_ids": dt_ids,
                    "tps": tps,
                    "fps": fps,
                }

                for iou_thr_idx, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    tp = np.array(tp)
                    fp = np.array(fp)
                    num_tp = len(tp)
                    rc = tp / num_gt
                    if num_tp:
                        recall[iou_thr_idx, cat_idx, area_idx] = rc[-1]
                    else:
                        recall[iou_thr_idx, cat_idx, area_idx] = 0

                    # np.spacing(1) ~= eps
                    pr = tp / (fp + tp + np.spacing(1))
                    pr = pr.tolist()

                    # Replace each precision value with the maximum precision
                    # value to the right of that recall level. This ensures
                    # that the  calculated AP value will be less suspectable
                    # to small variations in the ranking.
                    for i in range(num_tp - 1, 0, -1):
                        if pr[i] > pr[i - 1]:
                            pr[i - 1] = pr[i]

                    rec_thrs_insert_idx = np.searchsorted(rc,
                                                          self.params.rec_thrs,
                                                          side="left")

                    pr_at_recall = [0.0] * num_recalls

                    try:
                        for _idx, pr_idx in enumerate(rec_thrs_insert_idx):
                            pr_at_recall[_idx] = pr[pr_idx]
                    except BaseException:
                        pass
                    precision[iou_thr_idx, :, cat_idx,
                              area_idx] = np.array(pr_at_recall)

                # oLRP and Opt.Thr. Computation
                tp_num = np.cumsum(tps[0, :])
                # print(f"Shape of tp_num: {tp_num.shape}")
                fp_num = np.cumsum(fps[0, :])
                fn_num = num_gt - tp_num
                # If there is detection
                if tp_num.shape[0] > 0:
                    # There is some TPs
                    if tp_num[-1] > 0:
                        total_loc = tp_num - np.cumsum(dt_iou[0, :])
                        lrps = (total_loc / (1 - self.params.iou_thrs[0]) + fp_num + fn_num) / (tp_num + fp_num + fn_num)
                        _lrps[cat_idx, area_idx] = lrps
                        _dt_scores[cat_idx, area_idx] = dt_scores
                        _dt_m[cat_idx, area_idx] = dt_m
                        _tps[cat_idx, area_idx] = tps
                        _fps[cat_idx, area_idx] = fps
                        # print(f"Shape of lrps: {lrps.shape}")
                        opt_pos_idx = np.argmin(lrps)
                        olrp[cat_idx, area_idx] = lrps[opt_pos_idx]
                        olrp_loc[cat_idx, area_idx] = total_loc[opt_pos_idx] / tp_num[opt_pos_idx]
                        olrp_fp[cat_idx, area_idx] = fp_num[opt_pos_idx] / (tp_num[opt_pos_idx] + fp_num[opt_pos_idx])
                        olrp_fn[cat_idx,area_idx] = fn_num[opt_pos_idx] / num_gt
                        lrp_opt_thr[cat_idx, area_idx] = dt_scores[opt_pos_idx]
                    # There is No TP
                    else:
                        _lrps[cat_idx, area_idx] = None
                        _dt_scores[cat_idx, area_idx] = dt_scores
                        _dt_m[cat_idx, area_idx] = dt_m
                        _tps[cat_idx, area_idx] = tps
                        _fps[cat_idx, area_idx] = fps
                        olrp_loc[cat_idx, area_idx] = np.nan
                        olrp_fp[cat_idx, area_idx] = np.nan
                        olrp_fn[cat_idx, area_idx] = 1.
                        olrp[cat_idx, area_idx] = 1.
                        lrp_opt_thr[cat_idx, area_idx] = np.nan
                # No detection
                else:
                    olrp_loc[cat_idx, area_idx] = np.nan
                    olrp_fp[cat_idx, area_idx] = np.nan
                    olrp_fn[cat_idx, area_idx] = 1.
                    olrp[cat_idx, area_idx] = 1.
                    lrp_opt_thr[cat_idx, area_idx] = np.nan
                    _lrps[cat_idx, area_idx] = None
                    _dt_scores[cat_idx, area_idx] = None
                    _dt_m[cat_idx, area_idx] = None
                    _tps[cat_idx, area_idx] = None
                    _fps[cat_idx, area_idx] = None
        self.eval = {
            "params": self.params,
            "counts": [num_thrs, num_recalls, num_cats, num_area_rngs],
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": precision,
            "recall": recall,
            "dt_pointers": dt_pointers,
            'olrp_loc': olrp_loc,
            'olrp_fp': olrp_fp,
            'olrp_fn': olrp_fn,
            'olrp': olrp,
            'lrp_opt_thr': lrp_opt_thr,
            'lrp_values': _lrps,
            'dt_scores': _dt_scores,
            'dt_m': _dt_m,
            'tps': _tps,
            'fps': _fps,
        }

    def _summarize(self,
                   summary_type,
                   iou_thr=None,
                   area_rng="all",
                   freq_group_idx=None):
        aidx = [
            idx for idx, _area_rng in enumerate(self.params.area_rng_lbl)
            if _area_rng == area_rng
        ]

        if summary_type == 'ap':
            s = self.eval["precision"]
            if iou_thr is not None:
                tidx = np.where(iou_thr == self.params.iou_thrs)[0]
                s = s[tidx]
            if freq_group_idx is not None:
                s = s[:, :, self.freq_groups[freq_group_idx], aidx]
            else:
                s = s[:, :, :, aidx]
        elif summary_type == 'ar':
            s = self.eval["recall"]
            if iou_thr is not None:
                tidx = np.where(iou_thr == self.params.iou_thrs)[0]
                s = s[tidx]
            s = s[:, :, aidx]
        else:
            # # dimension of LRP: [KxAxM]
            if summary_type == 'oLRP':
                s = self.eval['olrp'][:, aidx]
            elif summary_type == 'oLRP_loc':
                s = self.eval['olrp_loc'][:, aidx]
            elif summary_type == 'oLRP_fp':
                s = self.eval['olrp_fp'][:, aidx]
            elif summary_type == 'oLRP_fn':
                s = self.eval['olrp_fn'][:, aidx]
            elif summary_type == 'LRP_thr':
                s = self.eval['lrp_opt_thr'][:, aidx].squeeze(axis=1)
                # Floor by using 3 decimal digits
                return np.round(s - 0.5 * 10**(-3), 3)
            if freq_group_idx is not None:
                s = s[self.freq_groups[freq_group_idx], :]

        idx = (~np.isnan(s))
        s = s[idx]
        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        return mean_s

    def summarize(self):
        """Compute and display summary metrics for evaluation results."""
        if not self.eval:
            raise RuntimeError("Please run accumulate() first.")

        max_dets = self.params.max_dets

        self.results["AP"] = self._summarize('ap')
        self.results["AP50"] = self._summarize('ap', iou_thr=0.50)
        self.results["AP75"] = self._summarize('ap', iou_thr=0.75)
        self.results["APs"] = self._summarize('ap', area_rng="small")
        self.results["APm"] = self._summarize('ap', area_rng="medium")
        self.results["APl"] = self._summarize('ap', area_rng="large")
        self.results["APr"] = self._summarize('ap', freq_group_idx=0)
        self.results["APc"] = self._summarize('ap', freq_group_idx=1)
        self.results["APf"] = self._summarize('ap', freq_group_idx=2)

        key = "AR@{}".format(max_dets)
        self.results[key] = self._summarize('ar')

        for area_rng in ["small", "medium", "large"]:
            key = "AR{}@{}".format(area_rng[0], max_dets)
            self.results[key] = self._summarize('ar', area_rng=area_rng)

        self.results["oLRP"] = self._summarize('oLRP')
        self.results["oLRP LOC"] = self._summarize('oLRP_loc')
        self.results["oLRP FP"] = self._summarize('oLRP_fp')
        self.results["oLRP FN"] = self._summarize('oLRP_fn')
        self.results["oLRPr"] = self._summarize('oLRP', freq_group_idx=0)
        self.results["oLRPc"] = self._summarize('oLRP', freq_group_idx=1)
        self.results["oLRPf"] = self._summarize('oLRP', freq_group_idx=2)
        self.results["lrp_opt_thr"] = self._summarize('LRP_thr')
        self.results["lrp_values"] = self.eval['lrp_values']
        self.results["dt_scores"] = self.eval['dt_scores']
        self.results["dt_m"] = self.eval['dt_m']
        self.results["tps"] = self.eval['tps']
        self.results["fps"] = self.eval['fps']

    def run(self):
        """Wrapper function which calculates the results."""
        self.evaluate()
        self.accumulate()
        self.summarize()

    def print_results(self, file_path = None):
        template = " {:<18} {} @[ IoU={:<9} | area={:>6s} | " + \
            "maxDets={:>3d} catIds={:>3s}] = {:0.3f}"

        for key, value in self.results.items():
            max_dets = self.params.max_dets
            if "AP" in key:
                title = "Average Precision"
                _type = "(AP)"
            elif "AR" in key:
                title = "Average Recall"
                _type = "(AR)"
            elif "oLRP" in key:
                title = "oLRP"
                if "LOC" in key:
                    _type = "Loc "
                elif "FP" in key:
                    _type = "FP  "
                elif "FN" in key:
                    _type = "FN  "
                else:
                    _type = "    "
                iou = "{:0.2f}".format(0.50)
            else:
                continue

            if len(key) > 2 and key[2].isdigit():
                iou_thr = (float(key[-2:]) / 100)
                iou = "{:0.2f}".format(iou_thr)
            elif "oLRP" not in key:
                iou = "{:0.2f}:{:0.2f}".format(self.params.iou_thrs[0],
                                               self.params.iou_thrs[-1])

            if len(key) > 2 and key[-1] in ["r", "c", "f"]:
                cat_group_name = key[-1]
            else:
                cat_group_name = "all"

            if len(key) > 2 and key[-1] in ["s", "m", "l"]:
                area_rng = key[-1]
            else:
                area_rng = "all"

            if file_path:
                with open(file_path, 'a') as out_file:
                    out_file.write(template.format(title, _type, iou, area_rng, max_dets,
                                    cat_group_name, value))
                    out_file.write('\n')
            else:
                print(
                    template.format(title, _type, iou, area_rng, max_dets,
                                    cat_group_name, value))

    def print_lrp_opt_thresholds(self):
        np.set_printoptions(threshold=np.inf)
        title = '# Class-specific LRP-Optimal Thresholds # \n'
        print(title, np.round(self.results["LRP Opt Thr"] - 0.5 * 10**(-3), 3))
        print("If LRP-Optimal Threshold of a class is: ")
        print("nan: NO True Positive from that class ")
        print("-1 : NO Ground Truth from that class")

    def get_results(self):
        if not self.results:
            self.logger.warn("results is empty. Call run().")
        return self.results


class Params:
    def __init__(self, iou_type):
        """Params for LVIS evaluation API."""
        self.img_ids = []
        self.cat_ids = []
        # np.arange causes trouble.  the data point on arange is slightly
        # larger than the true value
        self.iou_thrs = np.linspace(0.5,
                                    0.95,
                                    int(np.round((0.95 - 0.5) / 0.05)) + 1,
                                    endpoint=True)
        self.rec_thrs = np.linspace(0.0,
                                    1.00,
                                    int(np.round((1.00 - 0.0) / 0.01)) + 1,
                                    endpoint=True)
        self.max_dets = 300
        self.area_rng = [
            [0**2, 1e5**2],
            [0**2, 32**2],
            [32**2, 96**2],
            [96**2, 1e5**2],
        ]
        self.area_rng_lbl = ["all", "small", "medium", "large"]
        self.use_cats = 1
        # We bin categories in three bins based how many images of the training
        # set the category is present in.
        # r: Rare    :  < 10
        # c: Common  : >= 10 and < 100
        # f: Frequent: >= 100
        self.img_count_lbl = ["r", "c", "f"]
        self.iou_type = iou_type
