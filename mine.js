
$(document).ready(function () {
    console.log("HI!");

    const numIterations = 20000;
    const learningRate = 0.001;
    //past data
    const trainY = [ /*315.97,316.91,317.64,318.45,318.99,319.62,320.04,321.38,322.16,323.04,324.62,325.68,326.32,327.45,329.68,330.18,331.11,332.04,333.83,335.4,336.84, 338.75, 340.11, 341.45, 343.05, 344.65, 346.12, 347.42, 349.19, 351.57, 353.12, 354.39, 355.61, 356.45, 357.1, 358.83, 360.82, 362.61, 363.73, 366.7, 368.38,*/ 369.55, 371.14, 373.28, 375.8, 377.52, 379.8, 381.9, 383.79, 385.6, 387.43, 389.9, 391.65, 393.85, 396.52, 398.65, 400.83, 404.24, 406.55, 408.52];

    const predNum = trainY.length + 1;
    const trainX = [];

    for (i = 0; i < trainY.length; i++) {
        trainX.push(i);
    }

    const m = tf.variable(tf.scalar(Math.random()));
    const b = tf.variable(tf.scalar(Math.random()));

    //linear regression function
    function predict(x) {
        return tf.tidy(function () {
            return m.mul(x).add(b);
        });
    }

    function loss(prediction, labels) {
        //subtracts the two arrays & squares each element of the tensor then finds the mean. 
        const error = prediction.sub(labels).square().mean();
        return error;
    }


    $('#btn').click(function () {
        $('#msg').show();
        var promise1 = new Promise(function (resolve, reject) {
            console.log("Start");
            setTimeout(function () {
                const optimizer = tf.train.sgd(learningRate);
                for (let iter = 0; iter < numIterations; iter++) {
                    optimizer.minimize(function () {
                        const predsYs = predict(tf.tensor1d(trainX));
                        predsYs.print();
                        stepLoss = loss(predsYs, tf.tensor1d(trainY))
                        console.log(stepLoss.dataSync()[0])
                        return stepLoss;
                    });

                }



                

                resolve('foo');
                plot();
            }, 300);
        });

        promise1.then(function (value) {
            console.log(value);
            $('#msg').hide();
            // expected output: "foo"
        });

        console.log(promise1);
    });





    const predictionsBefore = predict(tf.tensor1d(trainX));

    async function plot() {

        let plotData = [];

        for (let i = 0; i < trainY.length; i++) {
            plotData.push({ x: trainX[i], y: trainY[i] });
        }

        console.log(plotData);

        var ctx = document.getElementById("myChart").getContext("2d");

        $('.prediction').html(predNum * m.dataSync()[0] + b.dataSync()[0]);
        var scatterChart = new Chart(ctx, {
            type: "line",
            data: {
                datasets: [
                    {
                        label: "Training Data",
                        showLine: false,
                        data: plotData,
                        fill: false
                    },
                    {
                        label: predNum * m.dataSync()[0] + b.dataSync()[0],//"Y = "+ m.dataSync()[0]+"X + " + b.dataSync()[0],
                        data: [
                            {
                                x: 0,
                                y: b.dataSync()[0]
                            },
                            {
                                x: predNum,
                                y: predNum * m.dataSync()[0] + b.dataSync()[0]
                            }
                        ],

                        // Changes this dataset to become a line
                        type: "line",
                        borderColor: "red",
                        fill: false
                    }
                ]
            },
            options: {
                scales: {
                    xAxes: [
                        {
                            type: "linear",
                            position: "bottom"
                        }
                    ]
                }
            }
        });
    }

    plot();

});