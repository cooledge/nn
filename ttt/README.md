# TicTacTopia

I wanted to write a neural net to play tic-tac-toe. It is pretty straightforward. The input is the set of moves and the current player. The output is which move to make. If the move is a winning move the output is 1.0. If the move blocks a winning move by the opposing player the outputs is 0.5. All other outputs are zero. For example


This worked pretty good. I had the neural net play itself for all possible games. There are nine since once trained the neural net is deterministic. Of the nine games eight were ties and one was not. But I expect all the games to be ties.


I added an incremental learning feature that happened after each game was played. The neural net kept track of the moves of each player. Then for the winner I trained the neural net on the winning moves to reinforce that. For the loser I trained the same neural net but used the negation of the loss function to un-reinforce those moves. The original model is this


model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

The inverse model is

def inverse_loss(y_true, y_pred, from_logits=False, axis=-1):
  return -keras.backend.categorical_crossentropy(y_true, y_pred, from_logits=from_logits, axis=axis)

  imodel = keras.Model(inputs=model.inputs, outputs=model.outputs)
  imodel.compile(optimizer=tf.keras.optimizers.Adam(), loss=inverse_loss, metrics=['accuracy'])


Then after each game if the game was not a tie

if winner != TIE:
  mx = model if winner == X else imodel
  mx.fit([np.array(choice_inputs_moves[0]), np.array(choice_inputs_players[0])], np.array(choice_outputs[0]))
  mo = model if winner == O else imodel
  mo.fit([np.array(choice_inputs_moves[1]), np.array(choice_inputs_players[1])], np.array(choice_outputs[1]))

The result was the neural net learned from its mistakes and every game became a tie. The Code is [here] (https://github.com/cooledge/nn/blob/master/ttt/ttt.py)
