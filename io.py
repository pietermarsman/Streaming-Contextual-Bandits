import numpy

__author__ = 'pieter'

TEAMID = "Team2"
TEAM_PASS = "d4363ba54b3106673e12a0647ea88af5"


def check_id(id, name):
    if not (0 <= id <= 10000):
        raise IndexError(str(name) + " should be between 0 and 10000")


def check_header(header):
    pass


def check_adtype(adtype):
    pass


def check_product_id(product_id):
    pass


def check_price(price):
    pass


def get_context(run_id, i):
    check_id(run_id, "runid")
    check_id(i, "i")
    pass


def propose_page(run_id, i, header=15, adtype="square", product_id=10, price=10.):
    check_id(run_id, "runid")
    check_id(i, "i")
    check_header(header)
    check_adtype(adtype)
    check_product_id(product_id)
    check_price(price)
    pass