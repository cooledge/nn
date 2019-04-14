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

def pick_move_nn(model, state, next_player):
  current_tie_prediction = 0.0
  current_win_prediction = 0.0
  current_tie_i = None
  current_win_i = None
  for i in range(len(state)):
    if state[i] == N: 
      next_state = [s for s in state]
      next_state[i] = next_player
      prediction = model.predict(np.array([state+next_state]))[0]

      if prediction[OUTCOME_WIN] > current_win_prediction:
        current_win_prediction = prediction[OUTCOME_WIN]
        current_win_i = i

      if prediction[OUTCOME_TIE] > current_tie_prediction:
        current_tie_prediction = prediction[OUTCOME_TIE]
        current_tie_i = i

  if current_win_i is None or current_tie_i is None:
    pdb.set_trace()
  if current_win_prediction > current_tie_prediction:
    return current_win_i
  else:
    return current_tie_i

def pick_move_rand(state):
  positions = [i for i,v in enumerate(state) if v == N]
  selection = random.randint(1, len(positions))-1
  return positions[selection]

def do_move(model, state, next_player):
  if model:
    move = pick_move_nn(model, state, next_player)
  else:
    move = pick_move_rand(state)
  state[move] = next_player

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

def get_game(model):
  next_player = random.randint(1,2)
  state = [N for i in range(9)]

  transitions_X = [] 
  transitions_O = []
  last_state = N
  while not calculate_winner(state):
    do_move(model, state, next_player)
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

def get_games(player, games, game = [[0 for _ in range(9)]], cells = [0,1,2,3,4,5,6,7,8]):
#def get_games(player, games, game = [[0 for _ in range(9)]], cells = [0,1]):

  if cells == []:
    games += [game]
    return

  if calculate_winner(game[-1]) is not None:
    games += [game]
    return

  for cell in cells:
    next_cells = cells.copy()
    next_cells.remove(cell)

    next_state = game[-1].copy()
    next_state[cell] = player
    next_game = game.copy()
    next_game += [next_state]

    get_games(other_player(player), games, next_game, next_cells)
    
def get_complete_game(game):
  next_player = X

  transitions_X = [] 
  transitions_O = []
  last_state = game[0]
  for state in game[1:]:
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

def generate_complete_data(model, n_games):
  data_moves = []
  data_outcomes = []
  games = []
  get_games(X, games)

  '''
   init:
    
   move 1/X
   
   move 2/O 

   move 3/X

   move 4/0:
      X  
       XO
        O
   move 5/X
     X  
      XO
     X O

   move 6/O
     X O
      XO
     X O
  
  state1 = [X, N, N, N, X, O, N, N, O]
  state2 = [X, N, N, N, X, O, X, N, O]
  state3 = [X, N, O, N, X, O, X, N, O]

  for game in games:
    if game[4:7] == [state1, state2, state3]:
      pdb.set_trace()
      games = [game]
      break
 
  state1 = [X, N, N, N, X, O, N, N, O]
  state2 = [X, N, N, N, X, O, X, N, O]
  state3 = [X, N, O, N, X, O, X, N, O]
  for game in games:
    if game[0:3] == [state1, state2, state3]:
      pdb.set_trace()

  pdb.set_trace()
  pdb.set_trace()
  
  move O: state1 = [X, N, N, N, X, O, N, N, O]
  move X: state2 = [X, N, N, N, X, O, X, N, O]
  move O: state3 = [X, N, O, N, X, O, X, N, O]
  pdb.set_trace()
  win, loss, tie = find_outcome(state2, state3)
  look for this game [..., state1, state2, state3, ...] in games and check for loss
  '''

  for game in games:
    winner, moves_x, moves_o = get_complete_game(game)
    get_data(winner, X, moves_x, data_moves, data_outcomes)
    get_data(winner, O, moves_o, data_moves, data_outcomes)

  return np.array(data_moves), np.array(data_outcomes)

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

def generate_data(model, n_games):
  data_moves = []
  data_outcomes = []
  for _ in range(n_games):
    winner, moves_x, moves_o = get_game(model)
    get_data(winner, X, moves_x, data_moves, data_outcomes)
    get_data(winner, O, moves_o, data_moves, data_outcomes)

  return np.array(data_moves), np.array(data_outcomes)

try:
  with open('data', 'rb') as data_file:
    data = pickle.load(data_file)
    training_moves = data['training_moves']
    training_outcomes = data['training_outcomes']
except:
  training_moves, training_outcomes = generate_complete_data(None, N_GAMES_1)
  #training_moves, training_outcomes = generate_data(None, N_GAMES_1)

  with open('data', 'wb') as data_file:
    pickle.dump({'training_moves': training_moves, 'training_outcomes': training_outcomes}, data_file)

'''
ndtm = []
for tm in training_moves:
  tm = list(tm)
  if tm not in ndtm:
    ndtm += [tm]
  else:
    pdb.set_trace()
    pdb.set_trace()

pdb.set_trace()
'''

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

'''
pdb.set_trace()
training_moves, training_outcomes = generate_data(model, N_GAMES_2)
data_stats(training_moves, training_outcomes)
model.fit(training_moves, training_outcomes, epochs=20, batch_size=20)
'''

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


'''
2: X   0/216/24 0.0030582600738853216/_/0.10766869783401489
    XO
     O

1: X   96/0/0 0.9655637145042419/_/0.10471079498529434
    XO
   X O

2: X O 
    XO
     O
'''

'''
state1 = [X, N, N, N, X, O, N, N, O]
state2 = [X, N, N, N, X, O, X, N, O]
state3 = [X, N, O, N, X, O, X, N, O]
pdb.set_trace()
win, loss, tie = find_outcome(state2, state3)
'''



play_game()
play_game()
play_game()

'''
training_odds_inputs = []
training_odds_outcomes = []
for tm, to in zip(training_moves, training_outcomes):
  prediction = model.predict(np.array([tm]))[0]
  training_odds_inputs.append(prediction)
  training_odds_outcomes.append(to)

picker_model = keras.Sequential()
picker_model.add(keras.layers.Dense(256))
# WIN/LOSS/TIE
picker_model.add(keras.layers.Dense(3))
picker_model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
picker_model.fit(np.array(training_odds_inputs), np.array(training_odds_outcomes), epochs=EPOCHS, batch_size=20)
'''

