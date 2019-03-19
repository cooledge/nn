import pdb
import random
import tensorflow as tf

N = None
X = 1
O = 2
TIE = 3

OUTCOME_WIN = 0
OUTCOME_LOSS = 1
OUTCOME_TIE = 2

dek select_move(state):
  positions = [i for i,v in enumerate(state) if v == None]
  selection = random.randint(1, len(positions))-1
  return positions[selection]

def game_get(game, i, j):
  game[i*3+j]

def calculate_winner(game):
  lines = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
  ];
  for i in range(len(lines)):
    a, b, c = lines[i]
    if game[a] is not None and game[a] == game[b] and game[a] == game[c]:
      return game[a]
  
  if all(sq is not None for sq in game):
    return TIE;

  return None

def get_game():
  next_player = random.randint(1,2)
  state = [None for i in range(9)]

  transitions_X = [] 
  transitions_O = []
  last_state = None
  while not calculate_winner(state):
    state[select_move(state)] = next_player
    if next_player == X:
      if last_state is not None:
        transitions_X.append(last_state + state)
      next_player = O
    else:
      if last_state is not None:
        transitions_O.append(last_state + state)
      next_player = X
    last_state = [] + state
  return (calculate_winner(state), transitions_X, transitions_O)

training_move = []
training_outcome = []

winner, t_x, t_o = get_game()

if winner == X:
  outcome = OUTCOME_WIN
else winner == Y:
  outcome = OUTCOME_LOSS
else:
  outcome = OUTCOME_TIE
training_move.append(transition)
training_outcome.append(outcome)

#print(select_move([None, None, None]))
#print(select_move([None, 1, None]))
#winner = calculate_winner([X, O, N, N, X, N, N, O, X])
print(get_game())
pdb.set_trace()
pdb.set_trace()

'''
  S1+S2 -> X O Tie

  What about input the difference?
'''

model = keras.Sequential()
# 3 == X O None
model.add(keras.layers.Embedding(3, 32))
model.add(keras.layers.Dense(128))
# 3 == X O Tie
model.add(keras.layers.Dense(3))

model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_move, training_outcome, epochs=20, batch_size=100)


