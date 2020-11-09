import React from 'react';
import { Fragment } from 'react';
import io from 'socket.io-client';


export default class Liveness extends React.Component {

    streamObject;

    render() {
        return (
            <Fragment>
                <div id="videoContainer">
                    <video autoPlay width="340px" height="240px" id="videoCap"></video>
                    <br />
                    <button id="capBtn">Capture</button>
                </div>
            </Fragment>
        );
    }

    componentDidMount() {
        const constraints = {
            video: true
        };

        let eyeOpen1 = false;
        let eyeClose1 = false;
        let eyeOpen2 = false;

        const video = document.querySelector('#videoCap');
        const canvas = document.createElement('canvas');
        const capBtn = document.getElementById("capBtn");

        navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
            console.log(stream);
            video.srcObject = stream
            this.streamObject = stream

            const socket = io('ws://127.0.0.1:5001');

            socket.on('connect', () => {
                console.log("In Connect Socket...")
                const interval = setInterval(() => {
                    console.log("In timeout....")
                    canvas.width = video.videoWidth * 0.5;
                    canvas.height = video.videoHeight * 0.5;
                    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
                    let imageRaw = canvas.toDataURL('image/png');
                    const base64Image = imageRaw.split(',')[1];
                    socket.emit('is_eye_open', { image: base64Image });
                }, 300);

                socket.on('is_eye_open_res', (d) => {
                    if (d === true) {
                        if (!eyeOpen1) eyeOpen1 = d;
                        if (!eyeOpen2 && eyeClose1) {
                            eyeOpen2 = d;
                            clearInterval(interval);
                            console.log("Blink Detected.....");
                            alert("Blink Detected");
                        }
                    }
                    else {
                        if (!eyeClose1 && eyeOpen1) eyeClose1 = !d;
                    }
                })
            });

        }).catch(err => console.log(err));





    }

    componentWillUnmount() {
        console.log("*****In unmount*****")
        this.streamObject.getTracks().forEach(function (track) {
            track.stop();
        });
    }
}