import ast


# returns names of entities
def transform_json(value):
    # nan values skipped
    if type(value) == float:
        return ""

    ret_value = ""
    help_list = ast.literal_eval(value)
    for x in help_list:
        ret_value = ret_value + x['name'].replace(" ", "") + " "
    return ret_value


# returns 6 top-billed cast members
def add_cast(value):
    if type(value) == float:
        return ""

    ret_value = ""
    help_list = ast.literal_eval(value)
    for x in help_list:
        if help_list.index(x) > 6:
            break
        ret_value = ret_value + x['name'].replace(" ", "") + " "
    return ret_value


# returns only writer(s) and director(s)
def add_crew(value):
    if type(value) == float:
        return ""

    ret_value = ""
    help_list = ast.literal_eval(value)
    for x in help_list:
        if not (x['department'] == "Writing" or x['job'] == "Director"):
            continue
        ret_value = ret_value + x['name'].replace(" ", "") + " "
    return ret_value


# returns a string that contains info about movie
def get_bag_of_words(movie, movie_credits, keywords):
    bag_of_words = ""

    for x in ["genres", "spoken_languages", "production_companies"]:
        bag_of_words = bag_of_words + transform_json(movie[x])

    bag_of_words = bag_of_words + add_cast(movie_credits['cast']) \
                                + add_crew(movie_credits['crew']) \
                                + transform_json(keywords['keywords'])
    if movie['adult']:
        bag_of_words = bag_of_words + "adult"

    return bag_of_words
