# Program to generate word counts and sentiment counts for Marlowe & Shakespeare
import parsers
import csv
from os import listdir

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Note: you only need the line below once
#nltk.download('vader_lexicon')

def generateWordCounts(marloweTexts, shakespeareTexts, outfile):
    # Generate the word counts for each author
    marlowe = parsers.countWordsMany(marloweTexts)
    shakespeare = parsers.countWordsMany(shakespeareTexts)

    computeTFIDF(marlowe)
    computeTFIDF(shakespeare)

    # Write out the CSV labeling the author in addition to the typical fields
    csvFile = open(outfile, "w")
    csvWriter = csv.writer(csvFile)

    # Write in the header row as requested
    csvWriter.writerow(['Text', 'Author', 'Word', 'Count', 'TFIDF'])
    writeCountData(marlowe, 'Marlowe', csvWriter)
    writeCountData(shakespeare, 'Shakespeare', csvWriter)

    # Close file
    csvFile.close()

def computeTFIDF(dict):
    wordCounts = {}

    # Compute the doc counts
    for file in dict:
        for word in dict[file]:
            if not word in wordCounts:
                wordCounts[word] = 1
            else:
                wordCounts[word] += 1

    # Compute the TFIDF
    for file in dict:
        for word in dict[file]:
            cnt = dict[file][word]
            dict[file][word] = {'count':cnt, 'tfidf': cnt/float(wordCounts[word])}

def writeCountData(dataset, author, csvWriter):
    # Iterate through the word count directory and add to the csv
    for file in dataset:
        for word in dataset[file]:
            csvWriter.writerow([file, author, word, dataset[file][word]['count'], dataset[file][word]['tfidf']])

generateWordCounts('./RawTexts/Marlowe', './RawTexts/Shakespeare', 'rawCounts.csv')

# Write out the sentiment for each text (adding year manually to the csv)
def generateSentiment(marloweTexts, shakespeareTexts, outfile):
    # Compute the sentiment scores for each text
    marlowe = computeSentiment(marloweTexts)
    shakespeare = computeSentiment(shakespeareTexts)
    print("data is " + str(marlowe))

    # Write out the corresponding dataset
    csvFile = open(outfile, "w")
    csvWriter = csv.writer(csvFile)
    csvWriter.writerow(['Text', 'Author', 'Positive', 'Negative', 'Neutral'])
    writeSentimentData(marlowe, 'Marlowe', csvWriter)
    writeSentimentData(shakespeare, 'Shakespeare', csvWriter)

    csvFile.close()

def writeSentimentData(dict, author, csvWriter):
    # Iterate through the sentiment directory and add to the csv
    for file in dict:
        csvWriter.writerow([file, author, dict[file]['pos'], dict[file]['neg'], dict[file]['neu']])

def computeSentiment(textDirectory):
    sentimentScores = {}
    sid = SentimentIntensityAnalyzer()
    for doc in listdir(textDirectory):
        text = open(textDirectory +"/" + doc).read()
        ss = sid.polarity_scores(text)
        sentimentScores[doc] = {'neg': ss['neg'], 'pos': ss['pos'], 'neu': ss['neu']}

    return sentimentScores

generateSentiment('./RawTexts/Marlowe', './RawTexts/Shakespeare', 'sentimentData.csv')
