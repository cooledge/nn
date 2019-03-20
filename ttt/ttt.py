import pdb
import random
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

N = 0
X = 1
O = 2
TIE = 3

OUTCOME_WIN = 0
OUTCOME_LOSS = 1
OUTCOME_TIE = 2

N_GAMES = 50000

def select_move(state):
  positions = [i for i,v in enumerate(state) if v == N]
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
    if game[a] is not N and game[a] == game[b] and game[a] == game[c]:
      return game[a]
  
  if all(sq is not N for sq in game):
    return TIE;

  return None

def get_game():
  next_player = random.randint(1,2)
  state = [N for i in range(9)]

  transitions_X = [] 
  transitions_O = []
  last_state = N
  while not calculate_winner(state):
    state[select_move(state)] = next_player
    if next_player == X:
      if last_state is not N:
        transitions_X.append(last_state + state)
      next_player = O
    else:
      if last_state is not N:
        transitions_O.append(last_state + state)
      next_player = X
    last_state = [] + state
  return (calculate_winner(state), transitions_X, transitions_O)

def other_player(player):
  if player == X:
    return O
  else:
    return X

def state_to_one_hot(state):
  one_hot = []
  for s in state:
    oh = [0, 0, 0]
    oh[s] = 1
    one_hot += oh
  return one_hot

def get_data(winner, current_player, transitions, data_moves, data_outcomes):
  if winner == current_player:
    outcome = OUTCOME_WIN
  elif winner == other_player(current_player):
    outcome = OUTCOME_LOSS
  else:
    outcome = OUTCOME_TIE
  #data_moves += [state_to_one_hot(t) for t in transitions]
  data_moves += transitions
  data_outcomes += [outcome] * len(transitions)

data_moves = []
data_outcomes = []
for _ in range(N_GAMES):
  winner, moves_x, moves_o = get_game()
  get_data(winner, X, moves_x, data_moves, data_outcomes)
  get_data(winner, O, moves_o, data_moves, data_outcomes)

#pdb.set_trace() 
#data_moves = data_moves[0:10]
#data_outcomes = data_outcomes[0:10]

training_moves = np.array(data_moves)
training_outcomes = np.array(data_outcomes)

'''
  S1+S2 -> X O Tie

  What about input the difference?
'''

model = keras.Sequential()
# 3 == X O None
model.add(keras.layers.Embedding(3, 32, input_length=2*9))

# Try 1
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256))
# 3 == X O Tie
model.add(keras.layers.Dense(3, activation='softmax'))

model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("training_moves: {0} training_outcomes {1}".format(training_moves.shape, training_outcomes.shape))
model.fit(training_moves, training_outcomes, epochs=20, batch_size=20)

def pick_move(state, player):
  current_tie_prediction = 0.0
  current_win_prediction = 0.0
  current_tie_next_state = []
  current_win_next_state = []
  for i in range(len(state)):
    if state[i] == N: 
      next_state = [s for s in state]
      next_state[i] = player
      prediction = model.predict(np.array([state+next_state]))[0]

      if prediction[OUTCOME_WIN] > current_win_prediction:
        current_win_prediction = prediction[OUTCOME_WIN]
        current_win_next_state = next_state

      if prediction[OUTCOME_TIE] > current_tie_prediction:
        current_tie_prediction = prediction[OUTCOME_TIE]
        current_tie_next_state = next_state

  return (current_tie_prediction, current_tie_next_state, current_win_prediction, current_win_next_state)

def state_to_char(state):
  if state == X:
    return "X"
  elif state == O:
    return "O"
  else:
    return " "

def state_to_line(state):
  state = [state_to_char(s) for s in state]
  return "{0}{1}{2}".format(state[0], state[1], state[2])

def play_game():
  state = [N]*9

  current_player = X
  while calculate_winner(state) is None:
    pick_move(state, X)
    tie_pred, tie_state, win_pred, win_state = pick_move(state, current_player)
    if win_pred > tie_pred:
      state = win_state
    else:
      state = tie_state
    print("{0}: {1}\n   {2}\n   {3}\n".format(current_player, state_to_line(state[0:3]), state_to_line(state[3:6]), state_to_line(state[6:9])))
    current_player = other_player(current_player)

play_game()
