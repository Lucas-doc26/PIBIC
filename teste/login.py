from flask import Blueprint, request, render_template

login = Blueprint("login", __name__, template_folder="templates")

users = {
    'admin': 'admin',
    'vitao1': 'vitaosenha',
    'vitao2': 'vitaosenha'
}

@login.route('/validated_user', methods=['POST'])
def validated_user():
    if request.method == 'POST':
        user = request.form['user']
        password = request.form['password']
        print(user, password)
        if user in users and users[user] == password:
            if users[user] == 'admin':
                return render_template('home.html', user=user)
            else:
                return render_template('home.html')
        else:
            return '<h1>Você não é o vitao!</h1>'
    else:
        return render_template('login.html')


@login.route('/register_user')
def register_user():
    return render_template("register_user.html")


@login.route('/add_user', methods=['GET', 'POST'])
def add_user():
    global users
    if request.method == 'POST':
        user = request.form['user']
        password = request.form['password']
    else:
        user = request.args.get('user', None)
        password = request.args.get('password', None)
    users[user] = password
    return render_template("list_users.html", devices=users)


@login.route('/list_users')
def list_users():
    global users
    return render_template("list_users.html", devices=users)


@login.route('/remove_user')
def remove_user():
    return render_template("remove_user.html", devices=users)


@login.route('/del_user', methods=['GET', 'POST'])
def del_user():
    global users
    if request.method == 'POST':
        user = request.form['user']
    else:
        user = request.args.get('user', None)
    users.pop(user)
    return render_template("list_users.html", devices=users)