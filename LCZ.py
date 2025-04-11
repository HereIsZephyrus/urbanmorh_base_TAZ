LCZ_name_mapper = {
    1: "Compact highrise",
    2: "Compact midrise",
    3: "Compact lowrise",
    4: "Open highrise",
    5: "Open midrise",
    6: "Open lowrise",
    7: "Lightweight lowrise",
    8: "Large lowrise",
    9: "Sparsely built",
    10: "Heavy industry",
    11: "Dense trees",
    12: "Scattered trees",
    13: "Bush, scrub",
    14: "Low plants",
    15: "Bare rock or paved",
    16: "Beaches, dunes or sands",
    17: "Water"
}

def convert_id_string_to_int(id_string):
    if (id_string.isdigit()):
        return int(id_string)
    identifier = id_string[-1] # last char
    if not identifier.isdigit():
        return 11 + int(ord(identifier) - ord('A'))
    else:
        return (int(identifier) + 9) % 10 + 1

def get_lcz_name(lcz_id):
    if isinstance(lcz_id, str):
        lcz_id = convert_id_string_to_int(lcz_id)
    return LCZ_name_mapper[lcz_id]

def is_water(lcz_id):
    if isinstance(lcz_id, str):
        lcz_id = convert_id_string_to_int(lcz_id)
    return lcz_id == 17

def is_industrial(lcz_id):
    if isinstance(lcz_id, str):
        lcz_id = convert_id_string_to_int(lcz_id)
    return lcz_id == 10

def is_artificial(lcz_id):
    if isinstance(lcz_id, str):
        lcz_id = convert_id_string_to_int(lcz_id)
    return lcz_id <= 10

def compact_level(lcz_id):
    if isinstance(lcz_id, str):
        lcz_id = convert_id_string_to_int(lcz_id)
    if not is_artificial(lcz_id):
        return 0
    elif lcz_id <= 3:
        return 3
    elif lcz_id <= 7:
        return 2
    return 1

def height_level(lcz_id):
    if isinstance(lcz_id, str):
        lcz_id = convert_id_string_to_int(lcz_id)
    if not is_artificial(lcz_id):
        return 0
    elif lcz_id in [3,6,7,8,9]:
        return 1
    elif lcz_id in [2,5,10]:
        return 2
    else:
        return 3