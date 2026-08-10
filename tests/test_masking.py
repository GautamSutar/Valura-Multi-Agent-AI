from takehome.masking import mask_bank_account, mask_pan


def test_mask_pan():
    assert mask_pan("BSNZA2249H") == "****249H"


def test_mask_bank_account():
    assert mask_bank_account("99936853430090") == "****0090"


def test_mask_short_value_does_not_crash():
    assert mask_pan("AB") == "****AB"
