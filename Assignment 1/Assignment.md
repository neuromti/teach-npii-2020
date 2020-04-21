# Assingment 1: Installation of python

## 1. Install Anaconda on your computer
- [on Windows](https://www.datacamp.com/community/tutorials/installing-anaconda-windows)
- [on Mac](https://www.datacamp.com/community/tutorials/installing-anaconda-mac-os-x)
- [on Ubuntu](https://wiki.ubuntuusers.de/Anaconda/)

## 2. Install Git and create a github account (if you do not have one yet)
- Install git on [any platform](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).
- Create a [github](https://github.com/) account.

## 3. Clone the course directory and create your own branch
- Open a Terminal or Command Prompt, use `cd /path/to/your/directory/` to navigate to the directory in which you want to download the course material. Then type
```
git clone https://github.com/translationalneurosurgery/teach-npii-2020.git
```
- Navigate into the cloned directory with `cd teach-npii-2020`
- Create your own [branch](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging) and navigate into this branch through:
```
git branch <your_branch_name>
git checkout <your_branch_name>
```
- Now you can modify any files in the directory. To tell git about new files that you created use
```
git add <name_of_new_file> 
```
To commit your changes use
```
git commit -a -m 'A commit message' 
```
- To upload your updates (e.g. to hand in your results for an assignment) use
```
git push -u origin <your_branch_name>
```
- To download any updates from the online repository (e.g., to get new assignments) use
```
git pull
```

## 4. Open Spyder through one of the following options
- Open the [Anaconda-Navigator](https://docs.anaconda.com/anaconda/user-guide/getting-started/#open-nav-win) and click on the Spyder launch button,
- or open an Anaconda Prompt (on Windows) or a Terminal (on Mac/Ubuntu) and type `spyder`.

## 5. Launch the file "hello_world.py"
In Spyder, open the file "hello_world.py".

Launch the file by clicking on the "Run file"-Button: ![Run](run.png).

You should see the following output in the Spyder Console:
```
Hello, world!
```

