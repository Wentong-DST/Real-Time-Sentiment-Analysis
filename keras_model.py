from keras.models import *
from keras.layers import *
from keras.datasets import mnist


def LSTM_Keras(args):
	inputs = Input(shape=(args.max_len, args.feature_dim))
	lstm_out = Bidirectional(LSTM(args.lstm_out, return_sequences = True), name='bilstm')(inputs)
	attention = Dense(1, activation='tanh')(lstm_out)
	attention = Flatten()(attention)
	attention = Activation('softmax')(attention)
	attention = RepeatVector(args.lstm_out*2)(attention)
	attention = Permute([2, 1], name='attetion_vec')(attention)
	attention_mul = multiply([lstm_out, attention], name='attention_mul')
	out = Flatten()(attention_mul)
	for i in args.mlp_hidden:
		out = Dense(i, activation='sigmoid')(out)
		out = Dropout(args.dropout)(out)
	output = Dense(args.num_classes, activation='sigmoid')(out)

	model = Model(inputs=inputs, outputs=output)

	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	return model

def CNN_LSTM(args):
	model = Sequential()
	# model.add(Embedding(args.vocab_size, args.embed_size, input_shape=(args.max_len,)))
	# model.add(Dropout(args.dropout))
	model.add(Conv1D(filters=args.kernel_num, kernel_size=3, padding='same', activation='relu', input_shape=(args.max_len, args.feature_dim)))
	model.add(MaxPooling1D(pool_size=args.pool_size))
	#model.add(Bidirectional(LSTM(args.lstm_out, return_sequences = True)))
	model.add(LSTM(args.lstm_out))
	#model.add(Flatten())
	model.add(Dense(args.num_classes, activation='sigmoid'))
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	return model
