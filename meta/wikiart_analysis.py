import json
from os.path import join
from glob import glob

def wikiart_analysis():
    in_folder = 'D:\CSC 249\wikiart-master\wikiart-saved\meta'
    
    artists = set()
    genres = set()
    styles = set()
    materials = set()
    techniques = set()

    total_works = 0

    has_artist = 0
    has_genre = 0
    has_style = 0
    has_material = 0
    has_technique = 0

    json_paths = glob(join(in_folder, '*.json'))
    num_artists = len(json_paths)

    for i, artist_json_path in enumerate(json_paths):
        completion = (i / num_artists) * 100
        load_bar = ('#' * int(completion)) + ('-' * int(100 - completion))
        print('{}/{} [{}] {:.2%}%'.format(i, num_artists, load_bar, completion / 100), end = '\r')


        with open(artist_json_path, encoding = 'utf-8') as fi:
            artist_json = json.loads(fi.read())
        
        for work in artist_json:
            total_works += 1

            if 'artistContentId' in work:
                art = work['artistContentId']
                if art:
                    has_artist += 1
                    artists.add(int(art))
            
            if 'genre' in work:
                gen = work['genre']
                if gen:
                    has_genre += 1
                    genres.add(gen)

            if 'style' in work:
                sty = work['style']
                if sty:
                    has_style += 1
                    styles.add(sty)

            if 'material' in work:
                mat = work['material']
                if mat:
                    has_material += 1
                    materials.add(mat)

            if 'technique' in work:
                tec = work['technique']
                if tec:
                    has_technique += 1
                    techniques.add(tec)
        
    analysis = {
        'N' : total_works,
        'artists' : artists,
        'has_artist' : has_artist,
        'genres' : genres,
        'has_genre' : has_genre,
        'styles' : styles,
        'has_style' : has_style,
        'materials' :  materials,
        'has_material' : has_material,
        'techniques' : techniques,
        'has_technique' : has_technique
    }
    
    return analysis
