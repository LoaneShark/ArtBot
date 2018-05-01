import csv
from collections import defaultdict

genres = defaultdict(int)
styles = defaultdict(int)
artists = defaultdict(int)

with open('train_info.csv', 'r', encoding = 'utf8') as fi:
    reader = csv.reader(fi)
    next(reader)

    for row in reader:
        artist = row[1]
        style = row[3]
        genre = row[4]

        artists[row[1]] += 1
        styles[row[3]] += 1
        genres[row[4]] += 1

print('{} artists'.format(len(artists)))
print('{} styles'.format(len(styles)))
print('{} genres'.format(len(genres)))

print(styles)
print(genres)