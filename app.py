
import flask
import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

# Load models at the top of the app to load into memory only one time
with open('models/loan_application_model_lr.pickle', 'rb') as f:
    clf_lr = pickle.load(f)

with open('models/knn_regression.pkl', 'rb') as f:
    knn = pickle.load(f)
    ss = StandardScaler()

# Mapping dictionaries
genders_to_int = {'MALE': 1, 'FEMALE': 0}
married_to_int = {'YES': 1, 'NO': 0}
education_to_int = {'GRADUATED': 1, 'NOT GRADUATED': 0}
dependents_to_int = {'0': 0, '1': 1, '2': 2, '3+': 3}
self_employment_to_int = {'YES': 1, 'NO': 0}
property_area_to_int = {'RURAL': 0, 'SEMIRURAL': 1, 'URBAN': 2}

app = flask.Flask(__name__, template_folder='templates')

@app.route('/')
def main():
    return flask.render_template('index.html')

@app.route('/report')
def report():
    return flask.render_template('report.html')

@app.route('/jointreport')
def jointreport():
    return flask.render_template('jointreport.html')

@app.route("/Loan_Application", methods=['GET', 'POST'])
def Loan_Application():
    if flask.request.method == 'GET':
        return flask.render_template('Loan_Application.html')
    
    if flask.request.method =='POST':
        # Get input
        genders_type = flask.request.form['genders_type']
        marital_status = flask.request.form['marital_status']
        dependents = flask.request.form['dependents']
        education_status = flask.request.form['education_status']
        self_employment = flask.request.form['self_employment']
        applicantIncome = float(flask.request.form['applicantIncome'])
        coapplicantIncome = float(flask.request.form['coapplicantIncome'])
        loan_amnt = float(flask.request.form['loan_amnt'])
        term_d = int(flask.request.form['term_d'])
        credit_history = int(flask.request.form['credit_history'])
        property_area = flask.request.form['property_area']

        # Create original output dict
        output_dict = {
            'Applicant Income': applicantIncome,
            'Co-Applicant Income': coapplicantIncome,
            'Loan Amount': loan_amnt,
            'Loan Amount Term': term_d,
            'Credit History': credit_history,
            'Gender': genders_type,
            'Marital Status': marital_status,
            'Education Level': education_status,
            'No of Dependents': dependents,
            'Self Employment': self_employment,
            'Property Area': property_area
        }

        x = np.zeros(21)
        x[0] = applicantIncome
        x[1] = coapplicantIncome
        x[2] = loan_amnt
        x[3] = term_d
        x[4] = credit_history

        print('------this is array data to predict-------')
        print('X = '+str(x))
        print('------------------------------------------')

        pred = clf_lr.predict([x])[0]
        if pred == 1:
            res = '🎊🎊Congratulations! your Loan Application has been Approved!🎊🎊'
        elif pred == 0:
            res = '😔😔Unfortunately, your Loan Application has been Denied😔😔'
        else:
            res = '😔😔Unfortunately, your Loan Application has been Denied😔😔 (Prediction inconclusive)'


        # Prepare data for rendering
        temp = pd.DataFrame(index=[1])
        temp['genders_type'] = genders_to_int.get(genders_type.upper(), 0)
        temp['marital_status'] = married_to_int.get(marital_status.upper(), 0)
        temp['dependents'] = dependents_to_int.get(dependents.upper(), 0)
        temp['education_status'] = education_to_int.get(education_status.upper(), 0)
        temp['self_employment'] = self_employment_to_int.get(self_employment.upper(), 0)
        temp['applicantIncome'] = applicantIncome
        temp['coapplicantIncome'] = coapplicantIncome
        temp['loan_amnt'] = loan_amnt
        temp['term_d'] = term_d
        temp['credit_history'] = credit_history
        temp['property_area'] = property_area_to_int.get(property_area.upper(), 0)

        temp['loan_amnt_log'] = np.log(temp['loan_amnt'])
        temp['Total_Income'] = temp['applicantIncome'] + temp['coapplicantIncome']
        temp['Total_Income_log'] = np.log(temp['Total_Income'])
        temp['EMI'] = temp['loan_amnt'] / temp['term_d']
        temp['Balance Income'] = temp['Total_Income'] - (temp['EMI'] * 1000)
        temp = temp.drop(['applicantIncome', 'coapplicantIncome', 'loan_amnt', 'term_d'], axis=1)

        return flask.render_template('Loan_Application.html', original_input=output_dict, result=res)

if __name__ == '__main__':
    app.run(debug=True)