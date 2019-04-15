import pdb

import random
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import argparse
import os

parser = argparse.ArgumentParser(description="Tic Tac Toe player")
parser.add_argument("--clean", action='store_true', default=False, help="regenerate data and weights")
args = parser.parse_args()

def safe_remove(fn):
  try:
    os.remove(fn)
  except FileNotFoundError:
    0

if args.clean:
  safe_remove("data")
  safe_remove("model")
  
N = 0
X = 1
O = 2
TIE = 3

OUTCOME_WIN = 0
OUTCOME_LOSS = 1
OUTCOME_TIE = 2

EPOCHS = 1

if False:
  N_GAMES_1 = 100
  N_GAMES_2 = 100
else:
  N_GAMES_1 = 50000
  N_GAMES_2 = 10000

def game_get(game, i, j):
  game[i*3+j]

def calculate_winner(state):
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
    if state[a] is not N and state[a] == state[b] and state[a] == state[c]:
      return state[a]
  
  if all(sq is not N for sq in state):
    return TIE;

  return None

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

# games state -> next states
def get_games(player, games, game = [[0 for _ in range(9)]], cells = [0,1,2,3,4,5,6,7,8]):
#def get_games(player, games, game = [[0 for _ in range(9)]], cells = [0,1]):

  if calculate_winner(game[-1]) is not None:
    return

  next_states = []
  for cell in cells:
    next_state = game[-1].copy()
    next_state[cell] = player
    next_states.append(next_state)

    next_cells = cells.copy()
    next_cells.remove(cell)
    next_game = game.copy()
    next_game += [next_state]

    get_games(other_player(player), games, next_game, next_cells)
  games += [(game[-1], player, next_states)]
    
def get_move_result(player, state):
  winner = calculate_winner(state)
  if winner == O or winner == X:
    if player == winner:
      return 1.0
    else:
      return 0.0
  elif winner == TIE:
      return 0.5
  else:
      return 0.5
  
  #return {X: 1.0, O: 0, TIE: 0.5, None: 0.5}[result]
  #return {OUTCOME_WIN: 1.0, OUTCOME_LOSS: 0, OUTCOME_TIE: 0.5, None: 0.5}[result]

# x: list of state+next_states
# y: one hot 1.0 if win, 0 if loose, 0.5 if tie
def generate_data(model, n_games):
  x = []
  y = []
  games = []
  get_games(X, games)

  for (state, player, next_states) in games:
    choices = []
    for next_state in next_states:
      choices += [[state, next_state]]
    #x += [np.concatenate([state]+next_states)]
    x.append(choices);
    y += [[get_move_result(player, next_state) for next_state in next_states]]

  return np.array(x), np.array(y)

try:
  with open('data', 'rb') as data_file:
    data = pickle.load(data_file)
    training_moves = data['training_moves']
    training_outcomes = data['training_outcomes']
except:
  training_moves, training_outcomes = generate_data(None, N_GAMES_1)

  with open('data', 'wb') as data_file:
    pickle.dump({'training_moves': training_moves, 'training_outcomes': training_outcomes}, data_file)

pdb.set_trace()

def data_stats(moves, outcomes):
  win, loss, tie = 0, 0, 0
  for outcome in outcomes:
    if outcome == OUTCOME_WIN:
      win += 1
    elif outcome == OUTCOME_LOSS:
      loss += 1
    else:
      tie += 1 
  total = win + loss + tie
  print("Data: {0}/{1}/{2}\n".format(win/total*100, loss/total*100, tie/total*100))

data_stats(training_moves, training_outcomes)

'''
  S1+S2 -> X O Tie

  What about input the difference?
'''

try: 
  model = keras.models.load_model('model')
except:
  model = keras.Sequential()
# 3 == X O None

# Try 1
  if False:
    model.add(keras.layers.Embedding(3, 32, input_length=2*9))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256))

# Try 2
  N_FILTERS = 64

  model.add(keras.layers.Embedding(3, 32, input_length=2*9))
# (18, 32)
  model.add(keras.layers.Reshape((2, 3, 3, 32)))
# (2, 9, 32)
  model.add(keras.layers.Conv3D(N_FILTERS, (1,3,3)))
# (2, 1, 1, 64)

  if True:
    model.add(keras.layers.Reshape((N_FILTERS*2,)))
  else:
    model.add(keras.layers.Reshape((2, N_FILTERS,)))
    model.add(keras.layers.GlobalAveragePooling1D())

# (128)
# 3 == X O Tie
#model.add(keras.layers.Dense(3, activation='softmax'))
  model.add(keras.layers.Dense(3, activation=tf.nn.sigmoid))

  model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  print("training_moves: {0} training_outcomes {1}".format(training_moves.shape, training_outcomes.shape))
  model.fit(training_moves, training_outcomes, epochs=EPOCHS, batch_size=20)

  model.save("model")

def find_outcome(state, next_state):
  win = 0
  loss = 0
  tie = 0

  expected_state = state+next_state
  for i, move in enumerate(training_moves):
    if np.array_equal(move, expected_state):
      if training_outcomes[i] == OUTCOME_WIN:
        win += 1
      elif training_outcomes[i] == OUTCOME_LOSS:
        loss += 1
      else:
        tie += 1
  #if win == 0 and loss == 0 and tie == 0:
    #pdb.set_trace()
  return (win, loss, tie)
        
def pick_move(state, player):
  current_tie_prediction = 0.0
  current_tie_loss_prediction = 1.0
  current_win_prediction = 0.0
  current_win_loss_prediction = 1.0
  current_tie_next_state = []
  current_win_next_state = []
  for i in range(len(state)):
    if state[i] == N: 
      next_state = [s for s in state]
      next_state[i] = player
      prediction = model.predict(np.array([state+next_state]))[0]

      if prediction[OUTCOME_LOSS] < current_win_loss_prediction:
        current_win_prediction = prediction[OUTCOME_WIN]
        current_win_loss_prediction = prediction[OUTCOME_LOSS]
        current_win_next_state = next_state

      if prediction[OUTCOME_LOSS] < current_tie_loss_prediction:
        current_tie_prediction = prediction[OUTCOME_TIE]
        current_tie_loss_prediction = prediction[OUTCOME_LOSS]
        current_tie_next_state = next_state
      '''
      if prediction[OUTCOME_WIN] > current_win_prediction and prediction[OUTCOME_LOSS] < current_win_loss_prediction:
        current_win_prediction = prediction[OUTCOME_WIN]
        current_win_loss_prediction = prediction[OUTCOME_LOSS]
        current_win_next_state = next_state

      if prediction[OUTCOME_TIE] > current_tie_prediction and prediction[OUTCOME_LOSS] < current_tie_loss_prediction:
        current_tie_prediction = prediction[OUTCOME_TIE]
        current_tie_loss_prediction = prediction[OUTCOME_LOSS]
        current_tie_next_state = next_state
      '''

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

def play_game(first_move_is_rand = False):
  state = [N]*9

  current_player = X
  while calculate_winner(state) is None:
    if first_move_is_rand:
      move = random.randint(1, len(state))-1
      next_state = [s for s in state]
      next_state[move] = current_player
       
      first_move_is_rand = false
      win_pred = 0
      tie_pred = 0
    else:
      tie_pred, tie_state, win_pred, win_state = pick_move(state, current_player)
      prev_state = state
      if win_pred > tie_pred:
        state = win_state
      else:
        state = tie_state
   
    win, loss, tie = find_outcome(prev_state, state)
    print("{0}: {1} {4}/{5}/{6} {7}/_/{8}\n   {2}\n   {3}\n".format(current_player, state_to_line(state[0:3]), state_to_line(state[3:6]), state_to_line(state[6:9]), win, loss, tie, win_pred, tie_pred))
   
    choices = {} 
    for i in range(len(prev_state)):
      tstate = prev_state.copy()
      if tstate[i] == N:
        tstate[i] = current_player
        choices[str(tstate)] = find_outcome(prev_state, tstate)

    print(choices)
    
    current_player = other_player(current_player)


play_game()
