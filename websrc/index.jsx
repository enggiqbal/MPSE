import ReactDOM from 'react-dom';
import React from 'react';

import { MpseForm } from './form';
export class ExpOutput extends React.Component {
    constructor(props) {
        super(props);
this.state={

};
    }

    submitQuery(event) {

        var bodyFormData = new FormData();
        console.log(bodyFormData);

        this.setState({
            loading: true,
            showEditModal: false,
            error: '',
        });

        axios.post('/run', {
            query: this.state.queryText
        })
            .then(this.onQueryLoad)
            .catch(this.onQueryError);
    }


    onQueryLoad(response) {

        this.setState({
            loading: false,
            hasResults: true,
            response: response,
        });
 console.log(response);
    }



    render() {
        return (
            <button type="button" className="btn btn-primary"  onClick={this.submitQuery}>Run MPSE</button>
       );
    }

}







ReactDOM.render(<MpseForm></MpseForm>, document.getElementById('app'));