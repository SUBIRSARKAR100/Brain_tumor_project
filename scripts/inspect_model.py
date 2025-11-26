from tensorflow import keras
m=keras.models.load_model('Model/model.keras')
print('LAYERS AND OUTPUT SHAPES:')
for i,l in enumerate(m.layers):
    try:
        print(i, l.name, getattr(l,'output_shape', None), type(l))
    except Exception as e:
        print(i, l.name, 'ERROR', e)

print('\nMODEL SUMMARY:')
m.summary()
