

def main():

    ###
    ## CREATING A DATAFRAME FROM THE 1950-2019 FILES THAT CONTAINS ATTRIBUTE VALUES ##
    ###
    # open the csv file with music from 1950 to 2019 that has the metadata tags

    # only keep these attributes: “dating”, “violence”, “romantic”, “obscene”, “sadness”, “danceability”, “energy”, “acousticness”, “night/time”, “movement/places”, “light/visual perception”, and “family/spiritual”

    # remove any songs released before 1980s


    ###
    ## CREATING A DATAFRAM FROM THE OTHER TWO CSV FILES, SONG LYRICS AND GENIUS LYRICS ##
    ###
    # open both csv files and make them pandas df objects

    # for the song_lyrics dataframe, remove BTS instances

    # for the genius_lyrics dataframe, remove any songs that aren't in English and that were released before 1980

    # also for the genius lyrics dataframe, remove all brackets that contain words in them from the lyrics (the lyrics have stuff like {chorus} or [verse] or whatever)

    # combine the dataframes, can remove any columns that aren't in both dataframes



if __name__ == "__main__":
    main()