from engine import load_model,manager
import mlflow
import pytest
import sys


# def test_model():
#     model = load_model()
    
#     assert len(model) == 3
#     assert isinstance(model,dict)
#     assert isinstance(model['kmean'],mlflow.pyfunc.PyFuncModel)

# @pytest.mark.skipif(sys.version_info < (3, 7), reason="requires python3.7 or higher")
# @pytest.mark.parametrize(
#     "driver_tip,distance,num_passengers,trip_duration,payment_method,rate_code,extra_charges,toll_amount",
#     [(3,5,9,4,4,76,5,10),
#     (6,8,1,3,2,40,2,9)]
#                         )
# def test_manger(driver_tip,distance,num_passengers,trip_duration,
#                 payment_method,rate_code,extra_charges,toll_amount):
            
#     value = {"driver_tip" : driver_tip
#         ,"distance" : distance
#         ,"num_passengers" : num_passengers
#         ,"trip_duration" : trip_duration
#         ,"payment_method" : payment_method
#         ,"rate_code":rate_code
#         ,"extra_charges" : extra_charges
#         ,"toll_amount" : toll_amount}
#     result = manager(value,load_model())
#     assert isinstance(result,float)

def test_sandeep():
    a = 2 
    assert a == 2
