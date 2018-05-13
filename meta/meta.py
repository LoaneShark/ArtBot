import csv
import sys
from collections import defaultdict
import json
from os.path import join
from glob import glob
sys.path.append('/home/sloane/ArtBot/meta/')

# kaggle
def kaggle_analysis(titles={},genres={},styles={},artists={},fi = open('/home/sloane/ArtBot/pytorch_gan/meta/kaggle_meta.csv', 'r')):

	reader = csv.reader(fi)
	next(reader)

	for row in reader:
		artist = row[1]
		style = row[3]
		genre = row[4]
		filename = "k_" + row[0]

		genres[filename] = genre
		styles[filename] = style
		artists[filename] = artist

		title = row[2]
		titles[filename] = title
	fi.close()

	return (titles, genres, styles, artists)

# wikiart
def wikiart_analysis(titles={},genres={},styles={},artists={}):
    in_folder = './wikiart_meta/'
    
    artists = set()
    genres = set()
    styles = set()
    total_works = 0

    has_artist = 0
    has_genre = 0
    has_style = 0

    json_paths = glob(join(in_folder, '*.json'))
    num_artists = len(json_paths)
    #print(num_artists)

    for i, artist_json_path in enumerate(json_paths):
        #completion = (i / num_artists) * 100
        #load_bar = ('#' * int(completion)) + ('-' * int(100 - completion))
        #print('{}/{} [{}] {:.2%}%'.format(i, num_artists, load_bar, completion / 100), end = '\r')

        with open(artist_json_path) as fi:
            artist_json = json.loads(fi.read())
        
        for work in artist_json:
            total_works += 1
            if i == 0:
            	print(work)

            filename = "w_" + work['contentId'] + ".png"


            if 'title' in work:
            	title = work['title']
            	titles[filename] = title

            if 'artistName' in work:
            	artist = work['artistName']
            	artists[filename] = artist

            #if 'artistContentId' in work:
                #art = work['artistContentId']
                #if art:
                    #has_artist += 1
                    #artists.add(int(art))
            
            if 'genre' in work:
                genre = work['genre']
                if genre:
                	genres[filename] = genre
                    #has_genre += 1
                    #genres.add(gen)

            if 'style' in work:
                style = work['style']
                if style:
                	styles[filename] = style
                    #has_style += 1
                    #styles.add(sty)
        
    #analysis = {
        #'N' : total_works,
        #'artists' : artists,
        #'has_artist' : has_artist,
        #'genres' : genres,
        #'has_genre' : has_genre,
        #'styles' : styles,
        #'has_style' : has_style,
        #'materials' :  materials,
        #'has_material' : has_material,
        #'techniques' : techniques,
        #'has_technique' : has_technique
    #}
    
	return (titles, genres, styles, artists)
