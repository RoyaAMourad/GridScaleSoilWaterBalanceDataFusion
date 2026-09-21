using System.Globalization;
using System.IO;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;

//discharge closure: Qobs is used as a npisy obs on all prior components 
//Each term is described by a distribution with a nonnegativity constraint.
// Qrain, Qesc, Qbf are treated as unknown random variables 
// while the other terms (--fixed-irr,//--fixed-dom, --fixed-ind) are not inferred but treated as uncertain quantities:
//m_obs - m_dom - m_ind, v_obs + v_dom + v_ind. 

//priors CSV columns: date, mu/sig rain, mu/sig irr (row 0 defines the shared
//prior), mu/sig dom, mu/sig ind, mu/sig esc, y_obs, v_obs, mu/sig bf

//--fixed-irr-systematic: Qirr is a time-invariant variable with mean zero and random error 5 mm/month; its mean is already
//subtracted from the observation, and since the monthly closure constraint contains no Qirr term, its closure is not impacted by the variance of Qirr
var irrSystematic = args.Contains("--fixed-irr-systematic");
var fixedIrr   = args.Contains("--fixed-irr") || irrSystematic;
var fixedDom   = args.Contains("--fixed-dom");
var fixedInd   = args.Contains("--fixed-ind");
var paths      = args.Where(a => !a.StartsWith("--")).ToArray();
var priorsPath = paths.Length > 0 ? paths[0] : "priors.csv";
var postsPath  = paths.Length > 1 ? paths[1] : "posteriors.csv";
var ci = CultureInfo.InvariantCulture;

//read all rows: the closure is solved jointly over the full record
var dates  = new List<string>();
var pRain  = new List<Gaussian>();
var pDom   = new List<Gaussian>();
var pInd   = new List<Gaussian>();
var pEsc   = new List<Gaussian>();
var pBf = new List<Gaussian>();
var pObs   = new List<Gaussian>();
double muIrr = double.NaN, sigIrr = double.NaN;

using (var reader = new StreamReader(priorsPath))
{
    reader.ReadLine();
    string? line;
    while ((line = reader.ReadLine()) != null)
    {
        var p = line.Split(',');
        dates.Add(p[0]);
        pRain.Add( new Gaussian(double.Parse(p[1],  ci), Math.Pow(double.Parse(p[2],  ci), 2)));
        if (double.IsNaN(muIrr)) { muIrr = double.Parse(p[3], ci); sigIrr = double.Parse(p[4], ci); }
        pDom.Add(  new Gaussian(double.Parse(p[5],  ci), Math.Pow(double.Parse(p[6],  ci), 2)));
        pInd.Add(  new Gaussian(double.Parse(p[7],  ci), Math.Pow(double.Parse(p[8],  ci), 2)));
        pEsc.Add(  new Gaussian(double.Parse(p[9],  ci), Math.Pow(double.Parse(p[10], ci), 2)));
        pObs.Add(  new Gaussian(double.Parse(p[11], ci), double.Parse(p[12], ci)));
        pBf.Add(new Gaussian(double.Parse(p[13], ci), Math.Pow(double.Parse(p[14], ci), 2)));
    }
}
int T = dates.Count;

//marginalize the fixed terms out of each month's observation
for (int i = 0; i < T; i++)
{
    double mShift = 0.0, vAdd = 0.0;
    if (fixedIrr) { mShift += muIrr; if (!irrSystematic) vAdd += sigIrr * sigIrr; }
    if (fixedDom) { mShift += pDom[i].GetMean(); vAdd += pDom[i].GetVariance(); }
    if (fixedInd) { mShift += pInd[i].GetMean(); vAdd += pInd[i].GetVariance(); }
    if (mShift != 0.0 || vAdd != 0.0)
        pObs[i] = new Gaussian(pObs[i].GetMean() - mShift, pObs[i].GetVariance() + vAdd);
}

//joint model over the inferred terms
var t = new Microsoft.ML.Probabilistic.Models.Range(T).Named("t");
var priorRain  = Variable.Observed(pRain.ToArray(),  t).Named("priorRain");
var priorDom   = Variable.Observed(pDom.ToArray(),   t).Named("priorDom");
var priorInd   = Variable.Observed(pInd.ToArray(),   t).Named("priorInd");
var priorEsc   = Variable.Observed(pEsc.ToArray(),   t).Named("priorEsc");
var priorBf = Variable.Observed(pBf.ToArray(), t).Named("priorBf");
var likObs     = Variable.Observed(pObs.ToArray(),   t).Named("likObs");

//non-fixed Qirr: one shared variable across all months (constant in time)
Variable<double>? qIrr = null;
if (!fixedIrr)
{
    qIrr = Variable<double>.Random(new Gaussian(muIrr, sigIrr * sigIrr)).Named("qIrr");
    Variable.ConstrainPositive(qIrr);
}

var qRain  = Variable.Array<double>(t).Named("qRain");
var qEsc   = Variable.Array<double>(t).Named("qEsc");
var qBf = Variable.Array<double>(t).Named("qBf");
var qDom   = fixedDom ? null : Variable.Array<double>(t).Named("qDom");
var qInd   = fixedInd ? null : Variable.Array<double>(t).Named("qInd");

using (Variable.ForEach(t))
{
    qRain[t]  = Variable<double>.Random(priorRain[t]);
    qEsc[t]   = Variable<double>.Random(priorEsc[t]);
    qBf[t] = Variable<double>.Random(priorBf[t]);
    Variable.ConstrainPositive(qRain[t]);
    Variable.ConstrainPositive(qEsc[t]);
    Variable.ConstrainPositive(qBf[t]);   //baseflow: Qbf >= 0

    var sumQ = qRain[t] + qEsc[t] + qBf[t];
    if (!fixedIrr) sumQ += qIrr!;
    if (!fixedDom)
    {
        qDom![t] = Variable<double>.Random(priorDom[t]);
        Variable.ConstrainPositive(qDom[t]);
        sumQ += qDom[t];
    }
    if (!fixedInd)
    {
        qInd![t] = Variable<double>.Random(priorInd[t]);
        Variable.ConstrainPositive(qInd[t]);
        sumQ += qInd[t];
    }
    Variable.ConstrainEqualRandom(sumQ, likObs[t]);   //soft equality constraint
}

var engine = new InferenceEngine { ShowProgress = false, NumberOfIterations = 200 };

var postRain  = engine.Infer<Gaussian[]>(qRain);
var postEsc   = engine.Infer<Gaussian[]>(qEsc);
var postBf = engine.Infer<Gaussian[]>(qBf);
var postDom   = fixedDom ? pDom.ToArray() : engine.Infer<Gaussian[]>(qDom);   //fixed: posterior = prior
var postInd   = fixedInd ? pInd.ToArray() : engine.Infer<Gaussian[]>(qInd);
var postIrr   = fixedIrr ? new Gaussian(muIrr, sigIrr * sigIrr)
                         : engine.Infer<Gaussian>(qIrr);

using var writer = new StreamWriter(postsPath);
writer.WriteLine("date,mean_rain,var_rain,mean_irr,var_irr,mean_dom,var_dom,mean_ind,var_ind,mean_esc,var_esc,mean_bf,var_bf");
for (int i = 0; i < T; i++)
{
    writer.WriteLine(string.Join(",", dates[i],
        postRain[i].GetMean().ToString(ci),  postRain[i].GetVariance().ToString(ci),
        postIrr.GetMean().ToString(ci),      postIrr.GetVariance().ToString(ci),
        postDom[i].GetMean().ToString(ci),   postDom[i].GetVariance().ToString(ci),
        postInd[i].GetMean().ToString(ci),   postInd[i].GetVariance().ToString(ci),
        postEsc[i].GetMean().ToString(ci),   postEsc[i].GetVariance().ToString(ci),
        postBf[i].GetMean().ToString(ci), postBf[i].GetVariance().ToString(ci)));
}
string mode(bool f) => f ? "fixed" : "inferred";
var irrMode = irrSystematic ? "fixed-systematic" : mode(fixedIrr);
Console.WriteLine($"discharge closure: {T} months  irr={irrMode}, dom={mode(fixedDom)}, ind={mode(fixedInd)}");
Console.WriteLine($"Qirr posterior = N({postIrr.GetMean().ToString("F4", ci)}, {postIrr.GetVariance().ToString("E3", ci)})");
Console.WriteLine($"wrote {T} rows to {postsPath}");
