import pickle
corpus = '''The cat is sleeping on the mat.
I like to eat apples and oranges.
She walked to the store yesterday.
He plays guitar in a band.
They are going on vacation next month.
We watched a movie last night.
It's raining outside right now.
The dog chased the ball across the field.
My favorite color is blue.
She is studying for her exams.
He cooks delicious meals every Sunday.
We went hiking in the mountains last weekend.
The children are playing in the park.
I can't believe it's already June.
The book on the table belongs to Mary.
He has a black car parked in the driveway.
She is wearing a red dress to the party.
They have been friends since childhood.
I usually wake up early in the morning.
We visited our grandparents last summer.
She speaks three languages fluently.
The train arrives at 7 o'clock every morning.
He bought a new phone yesterday.
The students are listening to the teacher attentively.
We are planning a surprise party for him.
She always exercises before breakfast.
I forgot to bring my umbrella today.
They celebrated their anniversary last week.
He likes to read books in his free time.
The restaurant serves delicious Italian food.
She plays piano beautifully.
We need to finish this project by Friday.
The weather is getting colder these days.
He fixed the broken window yesterday.
She is a talented artist.
They went shopping at the mall.
I need to buy groceries this evening.
The cat climbed up the tree quickly.
He found a job after graduation.
She wrote a letter to her friend.
'''
corpus = corpus.split('\n')
corpus.remove('')
data = []
voc = set()
for sentence in corpus:
    sentence = sentence.strip('.').replace("'", '')
    sentence = sentence.lower()
    words = sentence.split(' ')
    for word in words:
        voc.add(word)
    data.append(words)

with open(r'data/corpus.pkl', 'wb') as f:
    pickle.dump(data, f)
print("corpus creation done.")

with open(r'data/voc.pkl', 'wb') as f:
    pickle.dump(voc, f)
print("voc creation done.")
