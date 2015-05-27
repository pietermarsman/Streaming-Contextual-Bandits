import requests
import json

__author__ = 'pieter'

TEAMID = "Team2"
TEAM_PASS = "d4363ba54b3106673e12a0647ea88af5"

CONTEXT_URL = 'http://krabspin.uci.ru.nl/getcontext.json'
PROPOSE_PAGE_URL = 'http://krabspin.uci.ru.nl/proposePage.json'


def check_id(id, name):
    if not (0 <= id <= 10000):
        raise AttributeError(str(name) + " should be between 0 and 10000")


def check_header(header):
    correct_values = [5, 15, 35]
    if header not in correct_values:
        raise AttributeError("Header should be one of " + str(correct_values))


def check_adtype(adtype):
    correct_values = ["skyscraper", "square", "banner"]
    if adtype not in correct_values:
        raise AttributeError("Adtype should be one of " + str(correct_values))


def check_color(color):
    correct_values = ["green", "blue", "red", "black", "white"]
    if color not in correct_values:
        raise AttributeError("Color should be one of " + str(correct_values))


def check_product_id(product_id):
    if not (10 <= product_id <= 25):
        raise AttributeError("Productid should be between 10 and 25")


def check_price(price):
    if not (0. < price <= 50.):
        raise AttributeError("Price should be between 0 and 50")


def check_propose_result(result):
    error = result["effect"]["Error"] if "effect" in result and "Error" in result["effect"] else 1
    if error is not None:
        raise IOError("propose_page resulted in an error: " + str(result))


def get_context(run_id, i):
    check_id(run_id, "runid")
    check_id(i, "i")

    payload = {'i': i, 'runid': run_id, 'teamid': TEAMID, 'teampw': TEAM_PASS}
    ret = requests.get(CONTEXT_URL, params=payload)
    ret_dict = json.loads(ret.text)

    return ret_dict


def propose_page(run_id, i, header=15, adtype="square", productid=10, price=10., color='white'):
    check_id(run_id, "runid")
    check_id(i, "i")
    check_header(header)
    check_adtype(adtype)
    check_product_id(productid)
    check_price(price)
    check_color(color)

    payload = {'teamid': TEAMID, 'teampw': TEAM_PASS, 'runid': run_id, 'i': i, 'header': 5, 'adtype': adtype,
               'color': color, 'productid': productid, 'price': price}
    ret = requests.get(PROPOSE_PAGE_URL, params=payload)
    ret_dict = json.loads(ret.text)
    check_propose_result(ret_dict)

    return ret_dict

