use rand::Rng; // 0.8.5
use std::f64::consts;

// Why is this already better than the whole rest of my code gd.



#[derive(Copy,Clone,Debug)]
struct PreActivationVal {
    val: f64,
    activation_type: usize, //just use int for activation types for now at least
}
impl PreActivationVal {
    fn activate(&self) -> Option<(f64,f64)> {
        // returns (post_activation,post_prime_activation),
        // post prime needs to be recorded somewhere for the
        // backward pass
        return if self.activation_type.clone() == 0 { //0 is linear
            Option::Some((self.val,1.0))
        } else if self.activation_type.clone() == 1 { //1 is sigmoid
            let post_activation = 1.0 / (1.0 + consts::E.powf(-self.val));
            let post_prime_activation = consts::E.powf(self.val)
                / (1.0 + consts::E.powf(self.val)).powf(2.0);
            Option::Some((post_activation,post_prime_activation))
        } else {
            Option::None
        }
    }
}

fn forward_pass(cloned_layers: Vec<Vec<PreActivationVal>>,
                cloned_edge_maps: Vec<Vec<Vec<f64>>>,
                cloned_next_layer_biases: Vec<Vec<f64>>) -> (Vec<Vec<PreActivationVal>>,Vec<Vec<f64>>) {

    let mut post_prime_activations_by_layer: Vec<Vec<f64>> = vec![];
    let mut new_layers: Vec<Vec<PreActivationVal>> = vec![];
    new_layers.push(cloned_layers.get(0).unwrap().clone());

    for i in 0..(cloned_layers.len()-1) {
        let layer: Vec<PreActivationVal> = new_layers[i].clone();
        let mut edges: Vec<Vec<f64>> = cloned_edge_maps.get(i).unwrap().clone();
        let biases: &Vec<f64> = cloned_next_layer_biases.get(i).unwrap().as_ref();
        let next_layer_size = cloned_layers.get(i+1).unwrap().len().clone();
        println!("NxtLayerSz---------------------------{:?}",&next_layer_size);
        println!("LoopCnt---------------------------{:?}",&i);
        let result = forward_pass_on_layer(&layer,&mut edges,biases,&next_layer_size);

        new_layers.push(make_rand_inputs_from_f_vec(result.0, 1));
        let post_prime_activation_this_layer = result.1;
        &post_prime_activations_by_layer.push(post_prime_activation_this_layer.clone());
        dbg!(&new_layers);
        dbg!(&post_prime_activation_this_layer);
    }
    (new_layers,post_prime_activations_by_layer)
}

fn forward_pass_on_layer(node_start_values: &Vec<PreActivationVal>,
                         edge_map: &mut Vec<Vec<f64>>,
                         biases_of_next_layer: &Vec<f64>,
                         size_of_next_layer: &usize) -> (Vec<f64>,Vec<f64>) {
    let mut next_layer_values = vec![0f64; *size_of_next_layer];
    let mut this_layer_post_prime_values = vec![0f64; node_start_values.len()];
    for node_to in 0..*size_of_next_layer {
        let mut total_for_node_to = 0.0;
        let mut count = 0;
        let mut got_post_primes = false;
        for node in node_start_values.iter() {
            let pre_activation_val = node.activate().unwrap();
            total_for_node_to = total_for_node_to + (pre_activation_val.0 * edge_map[count][node_to]);

            if !got_post_primes {
                // I think this is right, double check whether post primes are stored at
                // the nodes, or in the nodes following or preceding
                this_layer_post_prime_values[count] = pre_activation_val.1;
            }

            count = count + 1;
        }
        got_post_primes = true;
        next_layer_values[node_to] = total_for_node_to + biases_of_next_layer[node_to];
        //dbg!(total_for_node_to);
    }
    (next_layer_values,this_layer_post_prime_values)
}

fn make_random_inputs(length: usize, act_type: usize) -> Vec<PreActivationVal> {
    let mut v: Vec<f64> = vec![];
    for i in 0..length {
        v.push(rand::thread_rng().gen_range(-1.0..1.0));
    }
    v.into_iter()
        .map(|x|PreActivationVal { val: x, activation_type: act_type }).collect()
}

fn make_rand_inputs_from_f_vec(f_vec: Vec<f64>, act_type: usize) -> Vec<PreActivationVal> {
    f_vec.into_iter()
        .map(|x|PreActivationVal { val: x, activation_type: act_type }).collect()
}
fn make_empty_inputs_of_len(length: usize, act_type: usize) -> Vec<PreActivationVal> {
    let mut f_vec = vec![];
    for i in 0..length {
        f_vec.push(0.0);
    }
    f_vec.into_iter()
        .map(|x|PreActivationVal { val: x, activation_type: act_type }).collect()
}


fn make_random_map(l0_size: usize, l1_size: usize) -> Vec<Vec<f64>> {
    let mut my_map: Vec<Vec<f64>> = vec![vec![0f64; l1_size]; l0_size];
    for x in 0..l1_size {
        for y in 0..l0_size {
            my_map[y][x] = rand::thread_rng().gen_range(-1.0..1.0);
        }
    }
    my_map
}

fn make_random_biases(length: usize) -> Vec<f64> {
    let mut following_layer_biases = vec![];
    for i in 0..length {
        following_layer_biases.push(rand::thread_rng().gen_range(-1.0..1.0));
    }
    following_layer_biases
}

fn make_randomly_initialized_nn_with_shape(shape: Vec<usize>) {
    let num_layers = *&shape.len();

    // pre-activation values, if input node then just from data, if hidden or output, then the sum
    // of incoming connection things
    let mut inputs = vec![];
    for i in 0..num_layers {
        if i == 0 {
            inputs.push(make_empty_inputs_of_len(shape[i], 0));
        }
        else {
            inputs.push(make_random_inputs(shape[i],1));
        }
    }

    let mut maps = vec![];
    for i in 0..(num_layers-1) {
        maps.push(make_random_map(shape[i],shape[i+1]));
    }

    let mut following_layer_biases = vec![];
    for i in 1..num_layers {
        following_layer_biases.push(make_random_biases(shape[i]));
    }

    let fp = forward_pass(inputs,maps,following_layer_biases);
    println!("{:?}",&fp.0[4].len());
    println!("{:?}",&fp.1);
}

fn main() {
    make_randomly_initialized_nn_with_shape(vec![6,4,5,2,7]);
}


/*
=============
THE DO IT DOC
=============

Just got preActivation with bias included, calculating post and postPrime activation vals
    along with representing edges between layers with an adjacency matrix. The postPrime
    isn't stored anywhere yet, but should be used to update an outside struct which can
    hold all the values necessary for backpropagation.

 // Layer sizes, honestly do I even need a way to define network architecture at runtime?
    // Really not sure I do, what purpose would that serve unless I was using networks with adaptive
    // architectures.
    let LI: usize = 6;
    let LH0: usize = 4;
    let LH1: usize = 5;
    let LH2: usize = 2;
    let LO: usize = 7;

    // pre-activation values, if input node then just from data, if hidden or output, then the sum
    // of incoming connection things
    let input: Vec<PreActivationVal> = make_random_inputs(LI, 0);
    let input2: Vec<PreActivationVal> = make_random_inputs(LH0, 1);
    let input3: Vec<PreActivationVal> = make_random_inputs(LH1, 1);
    let input4: Vec<PreActivationVal> = make_random_inputs(LH2, 1);
    let input5: Vec<PreActivationVal> = make_random_inputs(LO, 1);

    let my_map_0 = make_random_map(LI,LH0);
    let my_map_1 = make_random_map(LH0,LH1);
    let my_map_2 = make_random_map(LH1,LH2);
    let my_map_3 = make_random_map(LH2,LO);

    //let mut following_layer_biases_0 = make_random_biases(LI);
    let mut following_layer_biases_1 = make_random_biases(LH0);
    let mut following_layer_biases_2 = make_random_biases(LH1);
    let mut following_layer_biases_3 = make_random_biases(LH2);
    let mut following_layer_biases_4 = make_random_biases(LO);

    //dbg!(one_layer_fwd_pass_output);
    let fp = forward_pass(vec![input,input2,input3,input4,input5],vec![my_map_0,my_map_1,my_map_2,my_map_3], vec![following_layer_biases_1,following_layer_biases_2,following_layer_biases_3,following_layer_biases_4]);
    println!("{:?}",&fp.0[4].len());
    println!("{:?}",&fp.1);
 */