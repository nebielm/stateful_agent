from datetime import date, datetime


def parse_birthdate(birthdate_value, current_date=None) -> date:
    if isinstance(birthdate_value, datetime):
        birthdate = birthdate_value.date()
    elif isinstance(birthdate_value, date):
        birthdate = birthdate_value
    else:
        birthdate = datetime.fromisoformat(str(birthdate_value)).date()

    if current_date is None:
        today = datetime.now().date()
    elif isinstance(current_date, datetime):
        today = current_date.date()
    elif isinstance(current_date, date):
        today = current_date
    else:
        today = datetime.fromisoformat(str(current_date)).date()

    if birthdate > today:
        raise ValueError("Birthdate cannot be in the future")
    return birthdate


def calculate_age_from_birthdate(birthdate_value, current_date=None) -> int:
    today = current_date if current_date is not None else date.today()
    if isinstance(today, datetime):
        today = today.date()
    elif not isinstance(today, date):
        today = datetime.fromisoformat(str(today)).date()
    birthdate = parse_birthdate(birthdate_value, today)
    return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
