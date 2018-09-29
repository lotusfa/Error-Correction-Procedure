let learning_rate;
let test_case;
let init_weight = [];
let weights = [];
let sums = [];
let predicts = [];
let results = [];
let errors = [];
let num_of_generation = 0;
let score = 0;


init();

function init () {
  get_traning_set () ;
  initiator_mode ();
}

function update_init_weight () {
  test_case = JSON.parse(document.getElementById('test_case').value);
  let weight_length = test_case[0].length - 1;
  console.log(weight_length);
  init_weight = [];
  weights = [];
  for (var i = 0; i < weight_length; i++) {
    init_weight[i] = 0;
    if (i == weight_length - 1) { 
      document.getElementById('init_weight').value = JSON.stringify(init_weight); 
      weights[0] = init_weight;
    }
  }

}

function get_traning_set () {
  var xmlhttp = new XMLHttpRequest();
  xmlhttp.onreadystatechange = function() {
      if (this.readyState == 4 && this.status == 200) {
          var myObj = JSON.parse(this.responseText);
          learning_rate = myObj.learning_rate;
          test_case = myObj.data;
          init_input_value ();
          update_init_weight ();
          return myObj;
      }
  };

  xmlhttp.open("GET", "./training_set.json", true);
  xmlhttp.send();
}

function init_input_value () {
  document.getElementById('learning_rate').value = learning_rate;
  document.getElementById('test_case').value = JSON.stringify(test_case);
}

function update_value () {
  learning_rate = document.getElementById('learning_rate').value;
  test_case = JSON.parse(document.getElementById('test_case').value);
  init_weight = JSON.parse(document.getElementById('init_weight').value);
  weights = [];
  weights[0] = init_weight;
}

function update_statistic_tags () {
  document.getElementById('l_tag').innerHTML = learning_rate;
  document.getElementById('s_tag').innerHTML = score+"/"+test_case.length;
  document.getElementById('g_tag').innerHTML = num_of_generation;
  document.getElementById('w_tag').innerHTML = init_weight;
}

function run_simulation () {
  update_value ();
  simulator_mode ();
  init_html ();
}

function stop_simulation () {
  initiator_mode ();
}

function initiator_mode () {
    document.getElementById("initiator").style.display = "block";
    document.getElementById("simulator").style.display = "none";
}

function simulator_mode () {
    document.getElementById("initiator").style.display = "none";
    document.getElementById("simulator").style.display = "block";
}

function button_click () {
  if (stage == 0) {
    stage = 1;
    humans.sort ();
    document.getElementById("super_button").innerHTML = "Next";
  }else if (stage == 1) {
    stage = 0;
    humans.nextGeneration ();
    document.getElementById("super_button").innerHTML = "Sort";
  }
}

function init_html () {
  let r = document.getElementById('result');
  r.innerHTML = "";
  update_statistic_tags ();

  for (let i = 0; i < test_case.length ; i++) { 
    let t = document.createElement("tr");
    let t1 = document.createElement("td");
    let t2 = document.createElement("td");
    let t3 = document.createElement("td");
    let t4 = document.createElement("td");
    let t5 = document.createElement("td");
    let t6 = document.createElement("td");

    t.appendChild(t1);
    t.appendChild(t2);
    t.appendChild(t3);
    t.appendChild(t4);
    t.appendChild(t5);
    t.appendChild(t6);
    
    t.id = "data_" + i;

    t1.innerHTML = i;
    t2.innerHTML = test_case[i];
    t3.innerHTML = init_weight;
    t4.innerHTML = 0;
    t5.innerHTML = "true==true";
    t6.innerHTML = init_weight;
    t5.style = "color: green;";

    r.appendChild(t);
  }
}

function train (id) {
  if (id == 0) {
    num_of_generation++;
    weights[0] = init_weight;
    score = 0; 
  }

  let t = document.getElementById('data_'+id);
  t.children[2].innerHTML = weights[id];


  sum_of_(test_case[id],weights[id],(s)=>{
    sums[id] = s;
    t.children[3].innerHTML = sums[id];
    predicts[id] = new Boolean(sums[id]>=1);
    results[id] = (predicts[id]==test_case[id][test_case[id].length-1])?true:false;
    let e = 1 - sums[id];
    if (results[id]) {score++;}
    errors[id] = results[id]?0:e;

    t.children[4].innerHTML = predicts[id]+"=="+test_case[id][ test_case[id].length-1]+"|e:"+errors[id];
    t.children[4].style = results[id]?"color: green;":"color: red;";


    backpropagation(weights[id],test_case[id],errors[id],(w2)=>{
      t.children[5].innerHTML = w2;
      weights[id+1] = w2;
      if ( id < test_case.length - 1 ) { 
        id++;
        train (id); 
        return;
      }else{
        init_weight = weights[id+1];
        update_statistic_tags();
        return;
      }
    });
  });
}

function sum_of_ (d,w,callback){
  let sum = 0; 
  for (let i = 0; i < w.length ; i++) {
    sum += d[i]*w[i];
    if (i == w.length - 1 ) {

      callback( Math.round(sum * 100)/100 );
      return;
    }
  }
}

function backpropagation (w1,x,error,callback){
  let w2 = []; 
  for (let i = 0; i < w1.length ; i++) {
    let r = w1[i] + learning_rate*error*x[i];
    w2[i] = Math.round(r * 100)/100;

    if (i == w1.length - 1 ) {
      callback(w2);
      return;
    }
  }
}


