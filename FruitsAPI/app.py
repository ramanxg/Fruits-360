from flask import Flask
import pickle
import base64
from model import Model
from flask_restful import reqparse, abort, Api, Resource

app = Flask(__name__)
api = Api(app)


model = Model()

parser = reqparse.RequestParser()
parser.add_argument('imagebytes')

class Predict(Resource):
    def get(self):
        #parse arguments
        args = parser.parse_args()
        bytes_string = args['imagebytes']
        
        img = model.process(bytes_string)
        name, prob = model.predict(img)

        #create JSON
        output = {'fruit_name':  name, 'probability': str(prob)}

        return output

api.add_resource(Predict, '/')




if __name__ == "__main__":
    app.run()